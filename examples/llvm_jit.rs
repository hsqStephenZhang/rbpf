use std::collections::BTreeMap;
use std::io::Error;
use std::ptr;
use std::sync::atomic::AtomicUsize;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::ByteOrdering;
use inkwell::types::IntType;
use inkwell::values::{FunctionValue, IntValue, PointerValue};

use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};
use rbpf::assembler::assemble;
use rbpf::ebpf::{
    self, Insn, BPF_ALU_OP_MASK, BPF_IND, BPF_JEQ, BPF_JGE, BPF_JGT, BPF_JLE, BPF_JLT, BPF_JMP32,
    BPF_JNE, BPF_JSET, BPF_JSGE, BPF_JSGT, BPF_JSLE, BPF_JSLT, BPF_X,
};

#[allow(unused)]
struct EbpfTranslator<'a> {
    context: &'a Context,
    module: Module<'a>,
    builder: Builder<'a>,
    function: FunctionValue<'a>,
    registers: [PointerValue<'a>; 11], // R0-R10
    insn_blocks: BTreeMap<u32, BasicBlock<'a>>,
    insn_targets: BTreeMap<u32, (BasicBlock<'a>, BasicBlock<'a>)>,
    mem_start: PointerValue<'a>,
    mem_end: PointerValue<'a>,
    umem_start: PointerValue<'a>,
    umem_end: PointerValue<'a>,
    byte_ordering: ByteOrdering,
    intrinsics: [FunctionValue<'a>; 3],
    entry_block: BasicBlock<'a>,
}

const PROG_NAME: &str = "main";

fn gen_next_label(label_cnt: &AtomicUsize) -> String {
    let cnt = label_cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!("block_{}", cnt)
}

/// Analyze the program and build the CFG
///
/// We do this because cranelift does not allow us to switch back to a previously
/// filled block and add instructions to it. So we can't split the program as we
/// translate it.
fn build_cfg<'a>(
    insn_blocks: &mut BTreeMap<u32, BasicBlock<'a>>,
    insn_targets: &mut BTreeMap<u32, (BasicBlock<'a>, BasicBlock<'a>)>,
    ctx: &'a Context,
    prog: &[u8],
    function: FunctionValue<'a>,
    label_cnt: AtomicUsize,
) -> Result<(), Error> {
    let mut insn_ptr: usize = 0;
    while insn_ptr * ebpf::INSN_SIZE < prog.len() {
        let insn = ebpf::get_insn(prog, insn_ptr);

        match insn.opc {
            // This instruction consumes two opcodes
            ebpf::LD_DW_IMM => {
                insn_ptr += 1;
            }

            ebpf::JA
            | ebpf::JEQ_IMM
            | ebpf::JEQ_REG
            | ebpf::JGT_IMM
            | ebpf::JGT_REG
            | ebpf::JGE_IMM
            | ebpf::JGE_REG
            | ebpf::JLT_IMM
            | ebpf::JLT_REG
            | ebpf::JLE_IMM
            | ebpf::JLE_REG
            | ebpf::JNE_IMM
            | ebpf::JNE_REG
            | ebpf::JSGT_IMM
            | ebpf::JSGT_REG
            | ebpf::JSGE_IMM
            | ebpf::JSGE_REG
            | ebpf::JSLT_IMM
            | ebpf::JSLT_REG
            | ebpf::JSLE_IMM
            | ebpf::JSLE_REG
            | ebpf::JSET_IMM
            | ebpf::JSET_REG
            | ebpf::JEQ_IMM32
            | ebpf::JEQ_REG32
            | ebpf::JGT_IMM32
            | ebpf::JGT_REG32
            | ebpf::JGE_IMM32
            | ebpf::JGE_REG32
            | ebpf::JLT_IMM32
            | ebpf::JLT_REG32
            | ebpf::JLE_IMM32
            | ebpf::JLE_REG32
            | ebpf::JNE_IMM32
            | ebpf::JNE_REG32
            | ebpf::JSGT_IMM32
            | ebpf::JSGT_REG32
            | ebpf::JSGE_IMM32
            | ebpf::JSGE_REG32
            | ebpf::JSLT_IMM32
            | ebpf::JSLT_REG32
            | ebpf::JSLE_IMM32
            | ebpf::JSLE_REG32
            | ebpf::JSET_IMM32
            | ebpf::JSET_REG32
            | ebpf::EXIT
            | ebpf::TAIL_CALL => {
                prepare_jump_blocks(
                    insn_blocks,
                    insn_targets,
                    ctx,
                    insn_ptr,
                    &insn,
                    function,
                    &label_cnt,
                );
            }
            _ => {}
        }

        insn_ptr += 1;
    }

    Ok(())
}

fn prepare_jump_blocks<'a>(
    insn_blocks: &mut BTreeMap<u32, BasicBlock<'a>>,
    insn_targets: &mut BTreeMap<u32, (BasicBlock<'a>, BasicBlock<'a>)>,
    ctx: &'a Context,
    insn_ptr: usize,
    insn: &Insn,
    function: FunctionValue<'a>,
    label_cnt: &AtomicUsize,
) {
    let insn_ptr = insn_ptr as u32;
    let next_pc: u32 = insn_ptr + 1;
    let target_pc: u32 = (next_pc as isize + insn.off as isize).try_into().unwrap();

    // This is the fallthrough block
    let fallthrough_block = insn_blocks
        .entry(next_pc)
        .or_insert_with(|| ctx.append_basic_block(function, &gen_next_label(&label_cnt)))
        .clone();

    // Jump Target
    let target_block = insn_blocks
        .entry(target_pc)
        .or_insert_with(|| ctx.append_basic_block(function, &gen_next_label(&label_cnt)))
        .clone();

    // Mark the blocks for this instruction
    insn_targets.insert(insn_ptr, (fallthrough_block, target_block));
}

impl<'a> EbpfTranslator<'a> {
    fn new(context: &'a Context) -> Self {
        let module = context.create_module("ebpf_module");
        let byte_ordering = {
            let module = context.create_module("byte_ordering");
            let execution_engine = module
                .create_jit_execution_engine(OptimizationLevel::None)
                .unwrap();
            let target_data = execution_engine.get_target_data();
            target_data.get_byte_ordering()
        };
        let builder = context.create_builder();

        let i16_type = context.i16_type();
        let i32_type = context.i32_type();
        let i64_type = context.i64_type();
        let bswap16_function_type = i16_type.fn_type(&[i16_type.into()], false);
        let bswap16_intrinsic = module.add_function("llvm.bswap.i16", bswap16_function_type, None);
        let bswap32_function_type = i32_type.fn_type(&[i32_type.into()], false);
        let bswap32_intrinsic = module.add_function("llvm.bswap.i32", bswap32_function_type, None);
        let bswap64_function_type = i64_type.fn_type(&[i64_type.into()], false);
        let bswap64_intrinsic = module.add_function("llvm.bswap.i64", bswap64_function_type, None);

        // Define function type: i64 (i8*)
        let fn_type = context.i64_type().fn_type(
            &[
                // mem_start
                context.ptr_type(AddressSpace::default()).into(),
                // mem_len
                context.i64_type().into(),
                // umem_start
                context.ptr_type(AddressSpace::default()).into(),
                // umem_len
                context.i64_type().into(),
            ],
            false,
        );

        let function = module.add_function(PROG_NAME, fn_type, None);
        let entry_block = context.append_basic_block(function, "entry");

        builder.position_at_end(entry_block);

        let insn_blocks = BTreeMap::new();
        let insn_targets = BTreeMap::new();

        // Allocate registers R0-R10
        let mut registers = [None; 11];
        for (i, reg) in registers.iter_mut().enumerate() {
            *reg = Some(
                builder
                    .build_alloca(context.i64_type(), &format!("r{}", i))
                    .unwrap(),
            );
        }

        let mem_start = function.get_nth_param(0).unwrap().into_pointer_value();
        mem_start.set_name("mem_start");
        let mem_len = function.get_nth_param(1).unwrap().into_int_value();
        let mem_len = mem_len.const_to_pointer(context.ptr_type(AddressSpace::default()));
        let mem_end = builder
            .build_int_add(mem_start, mem_len, "mem_end")
            .unwrap();

        let umem_start = function.get_nth_param(2).unwrap().into_pointer_value();
        umem_start.set_name("umem_start");
        let umem_len = function.get_nth_param(3).unwrap().into_int_value();
        let umem_len = umem_len.const_to_pointer(context.ptr_type(AddressSpace::default()));
        let umem_end = builder
            .build_int_add(umem_start, umem_len, "umem_end")
            .unwrap();

        // Initialize other registers to 0
        let zero = context.i64_type().const_int(0, false);
        for i in 0..=10 {
            builder.build_store(registers[i].unwrap(), zero).unwrap();
        }
        builder
            .build_store(registers[1].unwrap(), mem_start)
            .unwrap();
        builder.build_store(registers[2].unwrap(), mem_end).unwrap();

        EbpfTranslator {
            context,
            module,
            builder,
            function,
            registers: registers.map(|x| x.unwrap()),
            insn_blocks,
            insn_targets,
            mem_start,
            mem_end,
            umem_start,
            umem_end,
            byte_ordering,
            intrinsics: [bswap16_intrinsic, bswap32_intrinsic, bswap64_intrinsic],
            entry_block,
        }
    }

    fn compile_function(&mut self, prog: &[u8]) {
        build_cfg(
            &mut self.insn_blocks,
            &mut self.insn_targets,
            self.context,
            prog,
            self.function,
            AtomicUsize::new(0),
        )
        .unwrap();

        self.translate_program(prog);
    }

    fn translate_program(&mut self, prog: &[u8]) {
        let builder = &self.builder;

        let mut insn_ptr: usize = 0;
        let mut prev_is_terminator = false;
        let mut current_block = self.entry_block;
        while insn_ptr * ebpf::INSN_SIZE < prog.len() {
            let insn = ebpf::get_insn(prog, insn_ptr);

            if let Some(block) = self.insn_blocks.get(&(insn_ptr as u32)) {
                if !prev_is_terminator && block != &current_block {
                    builder.build_unconditional_branch(*block).unwrap();
                    current_block = *block;
                }

                builder.position_at_end(*block);
            }
            prev_is_terminator = false;

            match insn.opc {
                ebpf::LD_ABS_B
                | ebpf::LD_ABS_H
                | ebpf::LD_ABS_W
                | ebpf::LD_ABS_DW
                | ebpf::LD_IND_B
                | ebpf::LD_IND_H
                | ebpf::LD_IND_W
                | ebpf::LD_IND_DW => {
                    let ty = match insn.opc {
                        ebpf::LD_ABS_B | ebpf::LD_IND_B => self.context.i8_type(),
                        ebpf::LD_ABS_H | ebpf::LD_IND_H => self.context.i16_type(),
                        ebpf::LD_ABS_W | ebpf::LD_IND_W => self.context.i32_type(),
                        ebpf::LD_ABS_DW | ebpf::LD_IND_DW => self.context.i64_type(),
                        _ => unreachable!(),
                    };

                    let mem_start = self.mem_start;
                    let offset = self
                        .context
                        .i64_type()
                        .const_int(insn.off as u64, false)
                        .const_to_pointer(self.context.ptr_type(AddressSpace::default()));
                    let addr = builder.build_int_add(mem_start, offset, "addr").unwrap();

                    // IND instructions additionally add the value of the source register
                    let is_ind = (insn.opc & BPF_IND) != 0;
                    let addr = if is_ind {
                        let src_reg = self.insn_src(&insn);
                        unsafe { addr.const_gep(ty, &[src_reg]) }
                    } else {
                        addr
                    };

                    let loaded = self.reg_load(ty, addr, 0);

                    let ext = if ty != self.context.i64_type() {
                        builder
                            .build_int_z_extend(loaded, self.context.i64_type(), "ext")
                            .unwrap()
                    } else {
                        loaded
                    };

                    self.set_dst(&insn, ext);
                }
                ebpf::LD_DW_IMM => {
                    insn_ptr += 1;
                    let next_insn = ebpf::get_insn(prog, insn_ptr);

                    let imm = (((insn.imm as u32) as u64) + ((next_insn.imm as u64) << 32)) as i64;
                    let iconst = self.context.i64_type().const_int(imm as u64, true);
                    self.set_dst(&insn, iconst);
                }
                // BPF_LDX class
                ebpf::LD_B_REG | ebpf::LD_H_REG | ebpf::LD_W_REG | ebpf::LD_DW_REG => {
                    let ty = match insn.opc {
                        ebpf::LD_B_REG => self.context.i8_type(),
                        ebpf::LD_H_REG => self.context.i16_type(),
                        ebpf::LD_W_REG => self.context.i32_type(),
                        ebpf::LD_DW_REG => self.context.i64_type(),
                        _ => unreachable!(),
                    };

                    let base = self
                        .insn_src(&insn)
                        .const_to_pointer(self.context.ptr_type(AddressSpace::default()));
                    let loaded = self.reg_load(ty, base, insn.off);

                    let ext = if ty != self.context.i64_type() {
                        builder
                            .build_int_z_extend(loaded, self.context.i64_type(), "ext")
                            .unwrap()
                    } else {
                        loaded
                    };

                    self.set_dst(&insn, ext);
                }
                // BPF_ST and BPF_STX class
                ebpf::ST_B_IMM
                | ebpf::ST_H_IMM
                | ebpf::ST_W_IMM
                | ebpf::ST_DW_IMM
                | ebpf::ST_B_REG
                | ebpf::ST_H_REG
                | ebpf::ST_W_REG
                | ebpf::ST_DW_REG => {
                    let ty = match insn.opc {
                        ebpf::ST_B_IMM | ebpf::ST_B_REG => self.context.i8_type(),
                        ebpf::ST_H_IMM | ebpf::ST_H_REG => self.context.i16_type(),
                        ebpf::ST_W_IMM | ebpf::ST_W_REG => self.context.i32_type(),
                        ebpf::ST_DW_IMM | ebpf::ST_DW_REG => self.context.i64_type(),
                        _ => unreachable!(),
                    };
                    let is_imm = match insn.opc {
                        ebpf::ST_B_IMM | ebpf::ST_H_IMM | ebpf::ST_W_IMM | ebpf::ST_DW_IMM => true,
                        ebpf::ST_B_REG | ebpf::ST_H_REG | ebpf::ST_W_REG | ebpf::ST_DW_REG => false,
                        _ => unreachable!(),
                    };

                    let value = if is_imm {
                        self.insn_imm(&insn)
                    } else {
                        self.insn_src(&insn)
                    };

                    let narrow = if ty != self.context.i64_type() {
                        builder.build_int_truncate(value, ty, "narrow").unwrap()
                    } else {
                        value
                    };

                    let base = self
                        .insn_dst(&insn)
                        .const_to_pointer(self.context.ptr_type(AddressSpace::default()));
                    self.reg_store(ty, base, insn.off, narrow);
                }

                ebpf::ST_W_XADD => unimplemented!(),
                ebpf::ST_DW_XADD => unimplemented!(),

                // BPF_ALU class
                // TODO Check how overflow works in kernel. Should we &= U32MAX all src register value
                // before we do the operation?
                // Cf ((0x11 << 32) - (0x1 << 32)) as u32 VS ((0x11 << 32) as u32 - (0x1 << 32) as u32
                ebpf::ADD32_IMM => {
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder.build_int_add(src, imm, "add32_imm").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::ADD32_REG => {
                    //((reg[_dst] & U32MAX) + (reg[_src] & U32MAX)) & U32MAX,
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder.build_int_add(lhs, rhs, "add32_reg").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::SUB32_IMM => {
                    // reg[_dst] = (reg[_dst] as i32).wrapping_sub(insn.imm)         as u64,
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder.build_int_sub(src, imm, "sub32_imm").unwrap();
                    self.set_dst32(&insn, res);
                }

                ebpf::SUB32_REG => {
                    // reg[_dst] = (reg[_dst] as i32).wrapping_sub(reg[_src] as i32) as u64,
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder.build_int_sub(lhs, rhs, "sub32_reg").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::MUL32_IMM => {
                    // reg[_dst] = (reg[_dst] as i32).wrapping_mul(insn.imm)         as u64,
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder.build_int_mul(src, imm, "mul32_imm").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::MUL32_REG => {
                    // reg[_dst] = (reg[_dst] as i32).wrapping_mul(reg[_src] as i32) as u64,
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder.build_int_mul(lhs, rhs, "mul32_reg").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::DIV32_IMM => {
                    // reg[_dst] = (reg[_dst] as u32 / insn.imm              as u32) as u64,
                    let res = if insn.imm == 0 {
                        self.context.i32_type().const_int(0, false)
                    } else {
                        let imm = self.insn_imm32(&insn);
                        let src = self.insn_dst32(&insn);
                        builder
                            .build_int_unsigned_div(src, imm, "div32_imm")
                            .unwrap()
                    };
                    self.set_dst32(&insn, res);
                }
                ebpf::DIV32_REG => {
                    // reg[_dst] = (reg[_dst] as u32 / reg[_src]             as u32) as u64,
                    let zero = self.context.i32_type().const_int(0, false);
                    let one = self.context.i32_type().const_int(1, false);

                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);

                    let rhs_is_zero = builder
                        .build_int_compare(inkwell::IntPredicate::EQ, rhs, zero, "rhs_is_zero")
                        .unwrap();
                    let safe_rhs = builder
                        .build_select(rhs_is_zero, one, rhs, "safe_rhs")
                        .unwrap();
                    let div_res = builder
                        .build_int_unsigned_div(lhs, safe_rhs.into_int_value(), "div32_reg")
                        .unwrap();

                    let res = builder
                        .build_select(rhs_is_zero, lhs, div_res, "res")
                        .unwrap();
                    self.set_dst32(&insn, res.into_int_value());
                }
                ebpf::OR32_IMM => {
                    // reg[_dst] = (reg[_dst] as u32             | insn.imm  as u32) as u64,
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder.build_or(src, imm, "or32_imm").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::OR32_REG => {
                    // reg[_dst] = (reg[_dst] as u32             | reg[_src] as u32) as u64,
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder.build_or(lhs, rhs, "or32_reg").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::AND32_IMM => {
                    // reg[_dst] = (reg[_dst] as u32             & insn.imm  as u32) as u64,
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder.build_and(src, imm, "and32_imm").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::AND32_REG => {
                    // reg[_dst] = (reg[_dst] as u32             & reg[_src] as u32) as u64,
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder.build_and(lhs, rhs, "and32_reg").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::LSH32_IMM => {
                    // reg[_dst] = (reg[_dst] as u32).wrapping_shl(insn.imm  as u32) as u64,
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder.build_left_shift(src, imm, "lsh32_imm").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::LSH32_REG => {
                    // reg[_dst] = (reg[_dst] as u32).wrapping_shl(reg[_src] as u32) as u64,
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder.build_left_shift(lhs, rhs, "lsh32_reg").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::RSH32_IMM => {
                    // reg[_dst] = (reg[_dst] as u32).wrapping_shr(insn.imm  as u32) as u64,
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder
                        .build_right_shift(src, imm, false, "rsh32_imm")
                        .unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::RSH32_REG => {
                    // reg[_dst] = (reg[_dst] as u32).wrapping_shr(reg[_src] as u32) as u64,
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder
                        .build_right_shift(lhs, rhs, false, "rsh32_reg")
                        .unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::NEG32 => {
                    // { reg[_dst] = (reg[_dst] as i32).wrapping_neg()                 as u64; reg[_dst] &= U32MAX; },
                    let src = self.insn_dst32(&insn);
                    let res = builder.build_int_neg(src, "neg32").unwrap();
                    // TODO: Do we need to mask the result?
                    self.set_dst32(&insn, res);
                }
                ebpf::MOD32_IMM => {
                    // reg[_dst] = (reg[_dst] as u32             % insn.imm  as u32) as u64,

                    if insn.imm != 0 {
                        let imm = self.insn_imm32(&insn);
                        let src = self.insn_dst32(&insn);
                        let res = builder
                            .build_int_unsigned_rem(src, imm, "mod32_imm")
                            .unwrap();
                        self.set_dst32(&insn, res);
                    }
                }
                ebpf::MOD32_REG => {
                    // reg[_dst] = (reg[_dst] as u32 % reg[_src]             as u32) as u64,
                    let zero = self.context.i32_type().const_int(0, false);
                    let one = self.context.i32_type().const_int(1, false);

                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);

                    let rhs_is_zero = builder
                        .build_int_compare(inkwell::IntPredicate::EQ, rhs, zero, "rhs_is_zero")
                        .unwrap();
                    let safe_rhs = builder
                        .build_select(rhs_is_zero, one, rhs, "safe_rhs")
                        .unwrap();
                    let div_res = builder
                        .build_int_unsigned_rem(lhs, safe_rhs.into_int_value(), "mod32_reg")
                        .unwrap();

                    let res = builder
                        .build_select(rhs_is_zero, lhs, div_res, "res")
                        .unwrap();
                    self.set_dst32(&insn, res.into_int_value());
                }
                ebpf::XOR32_IMM => {
                    // reg[_dst] = (reg[_dst] as u32             ^ insn.imm  as u32) as u64,
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder.build_xor(src, imm, "xor32_imm").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::XOR32_REG => {
                    // reg[_dst] = (reg[_dst] as u32             ^ reg[_src] as u32) as u64,
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder.build_xor(lhs, rhs, "xor32_reg").unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::MOV32_IMM => {
                    let imm = self.insn_imm32(&insn);
                    self.set_dst32(&insn, imm);
                }
                ebpf::MOV32_REG => {
                    // reg[_dst] = (reg[_src] as u32)                                as u64,
                    let src = self.insn_src32(&insn);
                    self.set_dst32(&insn, src);
                }
                ebpf::ARSH32_IMM => {
                    // { reg[_dst] = (reg[_dst] as i32).wrapping_shr(insn.imm  as u32) as u64; reg[_dst] &= U32MAX; },
                    let src = self.insn_dst32(&insn);
                    let imm = self.insn_imm32(&insn);
                    let res = builder
                        .build_right_shift(src, imm, true, "arsh32_imm")
                        .unwrap();
                    self.set_dst32(&insn, res);
                }
                ebpf::ARSH32_REG => {
                    // { reg[_dst] = (reg[_dst] as i32).wrapping_shr(reg[_src] as u32) as u64; reg[_dst] &= U32MAX; },
                    let lhs = self.insn_dst32(&insn);
                    let rhs = self.insn_src32(&insn);
                    let res = builder
                        .build_right_shift(lhs, rhs, true, "arsh32_reg")
                        .unwrap();
                    self.set_dst32(&insn, res);
                }

                ebpf::BE | ebpf::LE => {
                    let should_swap = match insn.opc {
                        ebpf::BE => self.byte_ordering == ByteOrdering::LittleEndian,
                        ebpf::LE => self.byte_ordering == ByteOrdering::BigEndian,
                        _ => unreachable!(),
                    };

                    if should_swap {
                        let swapped = match insn.imm {
                            16 => {
                                let bswap16 = self.intrinsics[0];
                                builder
                                    .build_call(
                                        bswap16,
                                        &[self.insn_dst16(&insn).into()],
                                        "bswap16",
                                    )
                                    .unwrap()
                                    .try_as_basic_value()
                                    .left()
                                    .unwrap()
                            }
                            32 => {
                                let bswap32 = self.intrinsics[1];
                                builder
                                    .build_call(
                                        bswap32,
                                        &[self.insn_dst32(&insn).into()],
                                        "bswap32",
                                    )
                                    .unwrap()
                                    .try_as_basic_value()
                                    .left()
                                    .unwrap()
                            }
                            64 => {
                                let bswap64 = self.intrinsics[2];
                                builder
                                    .build_call(bswap64, &[self.insn_dst(&insn).into()], "bswap64")
                                    .unwrap()
                                    .try_as_basic_value()
                                    .left()
                                    .unwrap()
                            }
                            _ => unreachable!(),
                        };

                        match insn.imm {
                            16 => self.set_dst_masked(&insn, swapped.into_int_value(), 0xffff),
                            32 => self.set_dst_masked(&insn, swapped.into_int_value(), 0xffffffff),
                            64 => self.set_dst(&insn, swapped.into_int_value()),
                            _ => unreachable!(),
                        }
                    }
                }

                // alu64 instructions
                ebpf::ADD64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder.build_int_add(src, imm, "add64_imm").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::ADD64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder.build_int_add(lhs, rhs, "add64_reg").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::SUB64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder.build_int_sub(src, imm, "sub64_imm").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::SUB64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder.build_int_sub(lhs, rhs, "sub64_reg").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::MUL64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder.build_int_mul(src, imm, "mul64_imm").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::MUL64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder.build_int_mul(lhs, rhs, "mul64_reg").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::DIV64_IMM => {
                    let res = if insn.imm == 0 {
                        self.context.i64_type().const_int(0, false)
                    } else {
                        let imm = self.insn_imm(&insn);
                        let src = self.insn_dst(&insn);
                        builder
                            .build_int_unsigned_div(src, imm, "div64_imm")
                            .unwrap()
                    };
                    self.set_dst(&insn, res);
                }
                ebpf::DIV64_REG => {
                    let zero = self.context.i64_type().const_int(0, false);
                    let one = self.context.i64_type().const_int(1, false);

                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);

                    let rhs_is_zero = builder
                        .build_int_compare(inkwell::IntPredicate::EQ, rhs, zero, "rhs_is_zero")
                        .unwrap();
                    let safe_rhs = builder
                        .build_select(rhs_is_zero, one, rhs, "safe_rhs")
                        .unwrap();
                    let div_res = builder
                        .build_int_unsigned_div(lhs, safe_rhs.into_int_value(), "div64_reg")
                        .unwrap();

                    let res = builder
                        .build_select(rhs_is_zero, lhs, div_res, "res")
                        .unwrap();
                    self.set_dst(&insn, res.into_int_value());
                }
                ebpf::OR64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder.build_or(src, imm, "or64_imm").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::OR64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder.build_or(lhs, rhs, "or64_reg").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::AND64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder.build_and(src, imm, "and64_imm").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::AND64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder.build_and(lhs, rhs, "and64_reg").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::LSH64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder.build_left_shift(src, imm, "lsh64_imm").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::LSH64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder.build_left_shift(lhs, rhs, "lsh64_reg").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::RSH64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder
                        .build_right_shift(src, imm, false, "rsh64_imm")
                        .unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::RSH64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder
                        .build_right_shift(lhs, rhs, false, "rsh64_reg")
                        .unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::NEG64 => {
                    let src = self.insn_dst(&insn);
                    let res = builder.build_int_neg(src, "neg32").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::MOD64_IMM => {
                    if insn.imm != 0 {
                        let imm = self.insn_imm(&insn);
                        let src = self.insn_dst(&insn);
                        let res = builder
                            .build_int_unsigned_rem(src, imm, "mod64_imm")
                            .unwrap();
                        self.set_dst(&insn, res);
                    }
                }
                ebpf::MOD64_REG => {
                    let zero = self.context.i64_type().const_int(0, false);
                    let one = self.context.i64_type().const_int(1, false);

                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);

                    let rhs_is_zero = builder
                        .build_int_compare(inkwell::IntPredicate::EQ, rhs, zero, "rhs_is_zero")
                        .unwrap();
                    let safe_rhs = builder
                        .build_select(rhs_is_zero, one, rhs, "safe_rhs")
                        .unwrap();
                    let div_res = builder
                        .build_int_unsigned_rem(lhs, safe_rhs.into_int_value(), "mod64_reg")
                        .unwrap();

                    let res = builder
                        .build_select(rhs_is_zero, lhs, div_res, "res")
                        .unwrap();
                    self.set_dst(&insn, res.into_int_value());
                }
                ebpf::XOR64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder.build_xor(src, imm, "xor64_imm").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::XOR64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder.build_xor(lhs, rhs, "xor64_reg").unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::MOV64_IMM => {
                    let imm = self.insn_imm(&insn);
                    self.set_dst(&insn, imm);
                }
                ebpf::MOV64_REG => {
                    let src = self.insn_src(&insn);
                    self.set_dst(&insn, src);
                }
                ebpf::ARSH64_IMM => {
                    let src = self.insn_dst(&insn);
                    let imm = self.insn_imm(&insn);
                    let res = builder
                        .build_right_shift(src, imm, true, "arsh64_imm")
                        .unwrap();
                    self.set_dst(&insn, res);
                }
                ebpf::ARSH64_REG => {
                    let lhs = self.insn_dst(&insn);
                    let rhs = self.insn_src(&insn);
                    let res = builder
                        .build_right_shift(lhs, rhs, true, "arsh64_reg")
                        .unwrap();
                    self.set_dst(&insn, res);
                }

                // BPF_JMP & BPF_JMP32 class
                ebpf::JA => {
                    let (_, target_block) = self.insn_targets[&(insn_ptr as u32)];

                    builder.build_unconditional_branch(target_block).unwrap();
                    prev_is_terminator = true;
                }

                ebpf::JEQ_IMM
                | ebpf::JEQ_REG
                | ebpf::JGT_IMM
                | ebpf::JGT_REG
                | ebpf::JGE_IMM
                | ebpf::JGE_REG
                | ebpf::JLT_IMM
                | ebpf::JLT_REG
                | ebpf::JLE_IMM
                | ebpf::JLE_REG
                | ebpf::JNE_IMM
                | ebpf::JNE_REG
                | ebpf::JSGT_IMM
                | ebpf::JSGT_REG
                | ebpf::JSGE_IMM
                | ebpf::JSGE_REG
                | ebpf::JSLT_IMM
                | ebpf::JSLT_REG
                | ebpf::JSLE_IMM
                | ebpf::JSLE_REG
                | ebpf::JSET_IMM
                | ebpf::JSET_REG
                | ebpf::JEQ_IMM32
                | ebpf::JEQ_REG32
                | ebpf::JGT_IMM32
                | ebpf::JGT_REG32
                | ebpf::JGE_IMM32
                | ebpf::JGE_REG32
                | ebpf::JLT_IMM32
                | ebpf::JLT_REG32
                | ebpf::JLE_IMM32
                | ebpf::JLE_REG32
                | ebpf::JNE_IMM32
                | ebpf::JNE_REG32
                | ebpf::JSGT_IMM32
                | ebpf::JSGT_REG32
                | ebpf::JSGE_IMM32
                | ebpf::JSGE_REG32
                | ebpf::JSLT_IMM32
                | ebpf::JSLT_REG32
                | ebpf::JSLE_IMM32
                | ebpf::JSLE_REG32
                | ebpf::JSET_IMM32
                | ebpf::JSET_REG32 => {
                    let (fallthrough, target) = self.insn_targets[&(insn_ptr as u32)];

                    let is_reg = (insn.opc & BPF_X) != 0;
                    let is_32 = (insn.opc & BPF_JMP32) != 0;
                    let op = match insn.opc {
                        c if (c & BPF_ALU_OP_MASK) == BPF_JEQ => IntPredicate::EQ,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JNE => IntPredicate::NE,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JGT => IntPredicate::UGT,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JGE => IntPredicate::UGE,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JLT => IntPredicate::ULT,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JLE => IntPredicate::ULE,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JSGT => IntPredicate::SGT,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JSGE => IntPredicate::SGE,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JSLT => IntPredicate::SLT,
                        c if (c & BPF_ALU_OP_MASK) == BPF_JSLE => IntPredicate::SLE,
                        // JSET is handled specially below
                        c if (c & BPF_ALU_OP_MASK) == BPF_JSET => IntPredicate::NE,
                        _ => unreachable!(),
                    };

                    let lhs = if is_32 {
                        self.insn_dst32(&insn)
                    } else {
                        self.insn_dst(&insn)
                    };
                    let rhs = match (is_reg, is_32) {
                        (true, false) => self.insn_src(&insn),
                        (true, true) => self.insn_src32(&insn),
                        (false, false) => self.insn_imm(&insn),
                        (false, true) => self.insn_imm32(&insn),
                    };

                    let cmp_res = if (insn.opc & BPF_ALU_OP_MASK) == BPF_JSET {
                        let jset_res = builder.build_and(lhs, rhs, "jset").unwrap();
                        builder
                            .build_int_compare(
                                inkwell::IntPredicate::NE,
                                jset_res,
                                lhs.get_type().const_int(0, false),
                                "jset_cond",
                            )
                            .unwrap()
                    } else {
                        builder.build_int_compare(op, lhs, rhs, "cmp_res").unwrap()
                    };
                    builder
                        .build_conditional_branch(cmp_res, target, fallthrough)
                        .unwrap();
                    prev_is_terminator = true;
                }
                ebpf::EXIT => {
                    let r0 = self
                        .builder
                        .build_load(self.context.i64_type(), self.registers[0], "ret_val")
                        .unwrap();
                    self.builder.build_return(Some(&r0)).unwrap();
                    prev_is_terminator = true;
                }
                ebpf::TAIL_CALL => unimplemented!(),
                _ => unimplemented!("inst: {:?}", insn),
            }
            insn_ptr += 1;
        }
    }

    fn reg_load<'b>(&self, ty: IntType<'b>, base: PointerValue<'b>, offset: i16) -> IntValue<'b>
    where
        'a: 'b,
    {
        // self.insert_bounds_check(bcx, ty, base, offset);
        // bcx.ins().load(ty, MemFlags::new(), base, offset as i32)
        if offset == 0 {
            return self
                .builder
                .build_load(ty, base, "loaded")
                .unwrap()
                .into_int_value();
        } else {
            let offset = ty
                .const_int(offset as u64, false)
                .const_to_pointer(self.context.ptr_type(AddressSpace::default()));
            let addr = self.builder.build_int_add(base, offset, "addr").unwrap();
            self.builder
                .build_load(ty, addr, "loaded")
                .unwrap()
                .into_int_value()
        }
    }

    fn reg_store(&self, ty: IntType<'_>, base: PointerValue<'_>, offset: i16, val: IntValue<'_>) {
        if offset == 0 {
            self.builder.build_store(base, val).unwrap();
        } else {
            let offset = ty
                .const_int(offset as u64, false)
                .const_to_pointer(self.context.ptr_type(AddressSpace::default()));
            let addr = self.builder.build_int_add(base, offset, "addr").unwrap();
            self.builder.build_store(addr, val).unwrap();
        }
    }

    fn insn_imm(&self, insn: &Insn) -> IntValue<'_> {
        self.context.i64_type().const_int(insn.imm as u64, false)
    }
    fn insn_imm32(&self, insn: &Insn) -> IntValue<'_> {
        self.context.i32_type().const_int(insn.imm as u64, false)
    }

    fn insn_dst(&self, insn: &Insn) -> IntValue<'_> {
        let dst = self.registers[insn.dst as usize];
        self.builder
            .build_load(self.context.i64_type(), dst, "dst_val")
            .unwrap()
            .into_int_value()
    }

    fn insn_dst32(&self, insn: &Insn) -> IntValue<'_> {
        let dst = self.registers[insn.dst as usize];
        self.builder
            .build_load(self.context.i32_type(), dst, "dst_val")
            .unwrap()
            .into_int_value()
    }

    fn insn_dst16(&self, insn: &Insn) -> IntValue<'_> {
        let dst = self.registers[insn.dst as usize];
        self.builder
            .build_load(self.context.i16_type(), dst, "dst_val")
            .unwrap()
            .into_int_value()
    }

    fn insn_src(&self, insn: &Insn) -> IntValue<'_> {
        let src = self.registers[insn.src as usize];
        self.builder
            .build_load(self.context.i64_type(), src, "src_val")
            .unwrap()
            .into_int_value()
    }
    fn insn_src32(&self, insn: &Insn) -> IntValue<'_> {
        let src = self.registers[insn.src as usize];
        self.builder
            .build_load(self.context.i32_type(), src, "src_val")
            .unwrap()
            .into_int_value()
    }

    fn set_dst(&self, insn: &Insn, val: IntValue<'_>) {
        let dst = self.registers[insn.dst as usize];
        self.builder.build_store(dst, val).unwrap();
    }

    fn set_dst_masked(&self, insn: &Insn, val: IntValue<'_>, mask: u64) {
        let dst = self.registers[insn.dst as usize];
        let mask = self.context.i64_type().const_int(mask, false);
        let val = self
            .builder
            .build_int_z_extend(val, self.context.i64_type(), "val")
            .unwrap();
        let masked_val = self.builder.build_and(val, mask, "masked_val").unwrap();
        self.builder.build_store(dst, masked_val).unwrap();
    }

    fn set_dst32(&self, insn: &Insn, val: IntValue<'_>) {
        let dst = self.registers[insn.dst as usize];
        let val32 = self
            .builder
            .build_and(
                val,
                self.context.i64_type().const_int(0xffffffff, false),
                "val32",
            )
            .unwrap();
        let val32 = self
            .builder
            .build_int_z_extend(val32, self.context.i64_type(), "val32_z_ext")
            .unwrap();

        self.builder.build_store(dst, val32).unwrap();
    }

    fn print_ir(&self) {
        self.module.print_to_stderr();
    }

    fn exec(&self, mem_ptr: *const u8, mem_len: u64) -> i64 {
        let ee = self
            .module
            .create_jit_execution_engine(inkwell::OptimizationLevel::None)
            .unwrap();
        let func = ee.get_function_address("main").unwrap();
        let func: extern "C" fn(*const u8, u64, *const u8, u64) -> i64 =
            unsafe { std::mem::transmute(func) };
        func(mem_ptr, mem_len, ptr::null(), 0)
    }
}

macro_rules! test_llvm {
    ($name:ident, $prog:expr, $expected:expr) => {
        fn $name() {
            let context = Context::create();
            let mut translator = EbpfTranslator::new(&context);

            let prog = assemble($prog).unwrap();
            let mem = &mut [];

            translator.compile_function(&prog);
            // translator.print_ir();
            let res: i64 = translator.exec(mem.as_ptr(), mem.len() as u64);
            if res as u64 != $expected {
                translator.print_ir();
            }
            assert_eq!(res as u64, $expected);
        }
    };
    ($name:ident, $prog:expr, $mem:expr, $expected:expr) => {
        fn $name() {
            let context = Context::create();
            let mut translator = EbpfTranslator::new(&context);

            let prog = assemble($prog).unwrap();
            let mem = &mut $mem;

            translator.compile_function(&prog);
            let res = translator.exec(mem.as_ptr(), mem.len() as u64);
            if res as u64 != $expected {
                translator.print_ir();
            }
            assert_eq!(res as u64, $expected);
        }
    };
}

macro_rules! test_llvm_raw {
    ($name:ident, $prog:expr, $expected:expr) => {
        fn $name() {
            let context = Context::create();
            let mut translator = EbpfTranslator::new(&context);

            let mem = &mut [];
            translator.compile_function(&prog);
            // translator.print_ir();
            let res: i64 = translator.exec(mem.as_ptr(), mem.len() as u64);
            if res as u64 != $expected {
                translator.print_ir();
            }
            assert_eq!(res as u64, $expected);
        }
    };
    ($name:ident, $prog:expr, $mem:expr, $expected:expr) => {
        fn $name() {
            let context = Context::create();
            let mut translator = EbpfTranslator::new(&context);

            let prog = $prog.as_ref();
            let mem = &mut $mem;

            translator.compile_function(&prog);
            let res = translator.exec(mem.as_ptr(), mem.len() as u64);
            if res as u64 != $expected {
                translator.print_ir();
            }
            assert_eq!(res as u64, $expected);
        }
    };
}

test_llvm!(
    test_llvm_add,
    "
    mov32 r0, 0
    mov32 r1, 2
    add32 r0, 1
    add32 r0, r1
    exit
    ",
    0x3
);

test_llvm!(
    test_llvm_alu64_arith,
    "
    mov r0, 0
    mov r1, 1
    mov r2, 2
    mov r3, 3
    mov r4, 4
    mov r5, 5
    mov r6, 6
    mov r7, 7
    mov r8, 8
    mov r9, 9
    add r0, 23
    add r0, r7
    sub r0, 13
    sub r0, r1
    mul r0, 7
    mul r0, r3
    div r0, 2
    div r0, r4
    exit
    ",
    0x2a
);

test_llvm!(
    test_llvm_alu64_bit,
    "
    mov r0, 0
    mov r1, 1
    mov r2, 2
    mov r3, 3
    mov r4, 4
    mov r5, 5
    mov r6, 6
    mov r7, 7
    mov r8, 8
    or r0, r5
    or r0, 0xa0
    and r0, 0xa3
    mov r9, 0x91
    and r0, r9
    lsh r0, 32
    lsh r0, 22
    lsh r0, r8
    rsh r0, 32
    rsh r0, 19
    rsh r0, r7
    xor r0, 0x03
    xor r0, r2
    exit
    ",
    0x11
);

test_llvm!(
    test_llvm_alu_arith,
    "
    mov32 r0, 0
    mov32 r1, 1
    mov32 r2, 2
    mov32 r3, 3
    mov32 r4, 4
    mov32 r5, 5
    mov32 r6, 6
    mov32 r7, 7
    mov32 r8, 8
    mov32 r9, 9
    add32 r0, 23
    add32 r0, r7
    sub32 r0, 13
    sub32 r0, r1
    mul32 r0, 7
    mul32 r0, r3
    div32 r0, 2
    div32 r0, r4
    exit
    ",
    0x2a
);

test_llvm!(
    test_llvm_alu_bit,
    "
    mov32 r0, 0
    mov32 r1, 1
    mov32 r2, 2
    mov32 r3, 3
    mov32 r4, 4
    mov32 r5, 5
    mov32 r6, 6
    mov32 r7, 7
    mov32 r8, 8
    or32 r0, r5
    or32 r0, 0xa0
    and32 r0, 0xa3
    mov32 r9, 0x91
    and32 r0, r9
    lsh32 r0, 22
    lsh32 r0, r8
    rsh32 r0, 19
    rsh32 r0, r7
    xor32 r0, 0x03
    xor32 r0, r2
    exit
    ",
    0x11
);

test_llvm!(
    test_llvm_arsh32_high_shift,
    "
    mov r0, 8
    lddw r1, 0x100000001
    arsh32 r0, r1
    exit
    ",
    0x4
);

test_llvm!(
    test_llvm_arsh,
    "
    mov32 r0, 0xf8
    lsh32 r0, 28
    arsh32 r0, 16
    exit
    ",
    0xffff8000
);

test_llvm!(
    test_llvm_arsh64,
    "
    mov32 r0, 1
    lsh r0, 63
    arsh r0, 55
    mov32 r1, 5
    arsh r0, r1
    exit
    ",
    0xfffffffffffffff8
);

test_llvm!(
    test_llvm_arsh_reg,
    "
    mov32 r0, 0xf8
    mov32 r1, 16
    lsh32 r0, 28
    arsh32 r0, r1
    exit
    ",
    0xffff8000
);

test_llvm!(
    test_llvm_be16_high,
    "
    ldxdw r0, [r1]
    be16 r0
    exit
    ",
    [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
    0x1122
);

test_llvm!(
    test_llvm_be32,
    "
    ldxw r0, [r1]
    be32 r0
    exit
    ",
    [0x11, 0x22, 0x33, 0x44],
    0x11223344
);

test_llvm!(
    test_llvm_be32_high,
    "
    ldxdw r0, [r1]
    be32 r0
    exit
    ",
    [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
    0x11223344
);

test_llvm!(
    test_llvm_be64,
    "
    ldxdw r0, [r1]
    be64 r0
    exit
    ",
    [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
    0x1122334455667788
);

test_llvm!(
    test_llvm_div32_imm,
    "
    lddw r0, 0x10000000c
    div32 r0, 4
    exit
    ",
    0x3
);

test_llvm!(
    test_llvm_div32_reg,
    "
    lddw r0, 0x10000000c
    mov r1, 4
    div32 r0, r1
    exit
    ",
    0x3
);

test_llvm!(
    test_llvm_div64_imm,
    "
    mov r0, 0xc
    lsh r0, 32
    div r0, 4
    exit
    ",
    0x300000000
);

test_llvm!(
    test_llvm_div64_reg,
    "
    mov r0, 0xc
    lsh r0, 32
    mov r1, 4
    div r0, r1
    exit
    ",
    0x300000000
);

test_llvm!(
    test_llvm_div64_by_zero_imm,
    "
    mov32 r0, 1
    div r0, 0
    exit
    ",
    0x0
);

test_llvm!(
    test_llvm_div_by_zero_imm,
    "
    mov32 r0, 1
    div32 r0, 0
    exit
    ",
    0x0
);

test_llvm!(
    test_llvm_mod64_by_zero_imm,
    "
    mov32 r0, 1
    mod r0, 0
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_mod_by_zero_imm,
    "
    mov32 r0, 1
    mod32 r0, 0
    exit
    ",
    0x1
);

// TODO: diff with cranelift
test_llvm!(
    test_llvm_div64_by_zero_reg,
    "
    mov32 r0, 1
    mov32 r1, 0
    div r0, r1
    exit
    ",
    0x1
);

// TODO: diff with cranelift
test_llvm!(
    test_llvm_div_by_zero_reg,
    "
    mov32 r0, 1
    mov32 r1, 0
    div32 r0, r1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_mod64_by_zero_reg,
    "
    mov32 r0, 1
    mov32 r1, 0
    mod r0, r1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_mod_by_zero_reg,
    "
    mov32 r0, 1
    mov32 r1, 0
    mod32 r0, r1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_exit,
    "
    mov r0, 0
    exit
    ",
    0x0
);

test_llvm!(
    test_llvm_ja,
    "
    mov r0, 1
    ja +1
    mov r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jeq_imm,
    "
    mov32 r0, 0
    mov32 r1, 0xa
    jeq r1, 0xb, +4
    mov32 r0, 1
    mov32 r1, 0xb
    jeq r1, 0xb, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jeq_reg,
    "
    mov32 r0, 0
    mov32 r1, 0xa
    mov32 r2, 0xb
    jeq r1, r2, +4
    mov32 r0, 1
    mov32 r1, 0xb
    jeq r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jge_imm,
    "
    mov32 r0, 0
    mov32 r1, 0xa
    jge r1, 0xb, +4
    mov32 r0, 1
    mov32 r1, 0xc
    jge r1, 0xb, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jle_imm,
    "
    mov32 r0, 0
    mov32 r1, 5
    jle r1, 4, +1
    jle r1, 6, +1
    exit
    jle r1, 5, +1
    exit
    mov32 r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jle_reg,
    "
    mov r0, 0
    mov r1, 5
    mov r2, 4
    mov r3, 6
    jle r1, r2, +2
    jle r1, r1, +1
    exit
    jle r1, r3, +1
    exit
    mov r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jgt_imm,
    "
    mov32 r0, 0
    mov32 r1, 5
    jgt r1, 6, +2
    jgt r1, 5, +1
    jgt r1, 4, +1
    exit
    mov32 r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jgt_reg,
    "
    mov r0, 0
    mov r1, 5
    mov r2, 6
    mov r3, 4
    jgt r1, r2, +2
    jgt r1, r1, +1
    jgt r1, r3, +1
    exit
    mov r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jlt_imm,
    "
    mov32 r0, 0
    mov32 r1, 5
    jlt r1, 4, +2
    jlt r1, 5, +1
    jlt r1, 6, +1
    exit
    mov32 r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jlt_reg,
    "
    mov r0, 0
    mov r1, 5
    mov r2, 4
    mov r3, 6
    jlt r1, r2, +2
    jlt r1, r1, +1
    jlt r1, r3, +1
    exit
    mov r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jit_bounce,
    "
    mov r0, 1
    mov r6, r0
    mov r7, r6
    mov r8, r7
    mov r9, r8
    mov r0, r9
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jne_reg,
    "
    mov32 r0, 0
    mov32 r1, 0xb
    mov32 r2, 0xb
    jne r1, r2, +4
    mov32 r0, 1
    mov32 r1, 0xa
    jne r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jset_imm,
    "
    mov32 r0, 0
    mov32 r1, 0x7
    jset r1, 0x8, +4
    mov32 r0, 1
    mov32 r1, 0x9
    jset r1, 0x8, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jset_reg,
    "
    mov32 r0, 0
    mov32 r1, 0x7
    mov32 r2, 0x8
    jset r1, r2, +4
    mov32 r0, 1
    mov32 r1, 0x9
    jset r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jsge_imm,
    "
    mov32 r0, 0
    mov r1, -2
    jsge r1, -1, +5
    jsge r1, 0, +4
    mov32 r0, 1
    mov r1, -1
    jsge r1, -1, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jsge_reg,
    "
    mov32 r0, 0
    mov r1, -2
    mov r2, -1
    mov32 r3, 0
    jsge r1, r2, +5
    jsge r1, r3, +4
    mov32 r0, 1
    mov r1, r2
    jsge r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jsle_imm,
    "
    mov32 r0, 0
    mov r1, -2
    jsle r1, -3, +1
    jsle r1, -1, +1
    exit
    mov32 r0, 1
    jsle r1, -2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jsle_reg,
    "
    mov32 r0, 0
    mov r1, -1
    mov r2, -2
    mov32 r3, 0
    jsle r1, r2, +1
    jsle r1, r3, +1
    exit
    mov32 r0, 1
    mov r1, r2
    jsle r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jsgt_imm,
    "
    mov32 r0, 0
    mov r1, -2
    jsgt r1, -1, +4
    mov32 r0, 1
    mov32 r1, 0
    jsgt r1, -1, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jsgt_reg,
    "
    mov32 r0, 0
    mov r1, -2
    mov r2, -1
    jsgt r1, r2, +4
    mov32 r0, 1
    mov32 r1, 0
    jsgt r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jslt_imm,
    "
    mov32 r0, 0
    mov r1, -2
    jslt r1, -3, +2
    jslt r1, -2, +1
    jslt r1, -1, +1
    exit
    mov32 r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jslt_reg,
    "
    mov32 r0, 0
    mov r1, -2
    mov r2, -3
    mov r3, -1
    jslt r1, r1, +2
    jslt r1, r2, +1
    jslt r1, r3, +1
    exit
    mov32 r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jeq32_imm,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0x0
    mov32 r1, 0xa
    jeq32 r1, 0xb, +5
    mov32 r0, 1
    mov r1, 0xb
    or r1, r9
    jeq32 r1, 0xb, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jeq32_reg,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 0xa
    mov32 r2, 0xb
    jeq32 r1, r2, +5
    mov32 r0, 1
    mov32 r1, 0xb
    or r1, r9
    jeq32 r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jge32_imm,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 0xa
    jge32 r1, 0xb, +5
    mov32 r0, 1
    or r1, r9
    mov32 r1, 0xc
    jge32 r1, 0xb, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jge32_reg,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 0xa
    mov32 r2, 0xb
    jge32 r1, r2, +5
    mov32 r0, 1
    or r1, r9
    mov32 r1, 0xc
    jge32 r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jgt32_imm,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 5
    or r1, r9
    jgt32 r1, 6, +4
    jgt32 r1, 5, +3
    jgt32 r1, 4, +1
    exit
    mov32 r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jgt32_reg,
    "
    mov r9, 1
    lsh r9, 32
    mov r0, 0
    mov r1, 5
    mov32 r1, 5
    or r1, r9
    mov r2, 6
    mov r3, 4
    jgt32 r1, r2, +4
    jgt32 r1, r1, +3
    jgt32 r1, r3, +1
    exit
    mov r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jle32_imm,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 5
    or r1, r9
    jle32 r1, 4, +5
    jle32 r1, 6, +1
    exit
    jle32 r1, 5, +1
    exit
    mov32 r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jle32_reg,
    "
    mov r9, 1
    lsh r9, 32
    mov r0, 0
    mov r1, 5
    mov r2, 4
    mov r3, 6
    or r1, r9
    jle32 r1, r2, +5
    jle32 r1, r1, +1
    exit
    jle32 r1, r3, +1
    exit
    mov r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jlt32_imm,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 5
    or r1, r9
    jlt32 r1, 4, +4
    jlt32 r1, 5, +3
    jlt32 r1, 6, +1
    exit
    mov32 r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jlt32_reg,
    "
    mov r9, 1
    lsh r9, 32
    mov r0, 0
    mov r1, 5
    mov r2, 4
    mov r3, 6
    or r1, r9
    jlt32 r1, r2, +4
    jlt32 r1, r1, +3
    jlt32 r1, r3, +1
    exit
    mov r0, 1
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jne32_imm,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 0xb
    or r1, r9
    jne32 r1, 0xb, +4
    mov32 r0, 1
    mov32 r1, 0xa
    or r1, r9
    jne32 r1, 0xb, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jne32_reg,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 0xb
    or r1, r9
    mov32 r2, 0xb
    jne32 r1, r2, +4
    mov32 r0, 1
    mov32 r1, 0xa
    or r1, r9
    jne32 r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jset32_imm,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 0x7
    or r1, r9
    jset32 r1, 0x8, +4
    mov32 r0, 1
    mov32 r1, 0x9
    jset32 r1, 0x8, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jset32_reg,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, 0x7
    or r1, r9
    mov32 r2, 0x8
    jset32 r1, r2, +4
    mov32 r0, 1
    mov32 r1, 0x9
    jset32 r1, r2, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_jsge32_imm,
    "
    mov r9, 1
    lsh r9, 32
    mov32 r0, 0
    mov32 r1, -2
    or r1, r9
    jsge32 r1, -1, +5
    jsge32 r1, 0, +4
    mov32 r0, 1
    mov r1, -1
    jsge32 r1, -1, +1
    mov32 r0, 2
    exit
    ",
    0x1
);

test_llvm!(
    test_llvm_lddw,
    "
    lddw r0, 0x1122334455667788
    exit
    ",
    0x1122334455667788
);

test_llvm!(
    test_llvm_lddw2,
    "
    lddw r0, 0x0000000080000000
    exit
    ",
    0x80000000
);

test_llvm!(
    test_llvm_ldxb_all,
    "
    mov r0, r1
    ldxb r9, [r0+0]
    lsh r9, 0
    ldxb r8, [r0+1]
    lsh r8, 4
    ldxb r7, [r0+2]
    lsh r7, 8
    ldxb r6, [r0+3]
    lsh r6, 12
    ldxb r5, [r0+4]
    lsh r5, 16
    ldxb r4, [r0+5]
    lsh r4, 20
    ldxb r3, [r0+6]
    lsh r3, 24
    ldxb r2, [r0+7]
    lsh r2, 28
    ldxb r1, [r0+8]
    lsh r1, 32
    ldxb r0, [r0+9]
    lsh r0, 36
    or r0, r1
    or r0, r2
    or r0, r3
    or r0, r4
    or r0, r5
    or r0, r6
    or r0, r7
    or r0, r8
    or r0, r9
    exit
    ",
    [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09],
    0x9876543210
);

test_llvm!(
    test_llvm_ldxb,
    "
    ldxb r0, [r1+2]
    exit
    ",
    [0xaa, 0xbb, 0x11, 0xcc, 0xdd],
    0x11
);

test_llvm!(
    test_llvm_ldxdw,
    "
    ldxdw r0, [r1+2]
    exit
    ",
    [0xaa, 0xbb, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0xcc, 0xdd],
    0x8877665544332211
);

test_llvm!(
    test_llvm_ldxh_all,
    "
    mov r0, r1
    ldxh r9, [r0+0]
    be16 r9
    lsh r9, 0
    ldxh r8, [r0+2]
    be16 r8
    lsh r8, 4
    ldxh r7, [r0+4]
    be16 r7
    lsh r7, 8
    ldxh r6, [r0+6]
    be16 r6
    lsh r6, 12
    ldxh r5, [r0+8]
    be16 r5
    lsh r5, 16
    ldxh r4, [r0+10]
    be16 r4
    lsh r4, 20
    ldxh r3, [r0+12]
    be16 r3
    lsh r3, 24
    ldxh r2, [r0+14]
    be16 r2
    lsh r2, 28
    ldxh r1, [r0+16]
    be16 r1
    lsh r1, 32
    ldxh r0, [r0+18]
    be16 r0
    lsh r0, 36
    or r0, r1
    or r0, r2
    or r0, r3
    or r0, r4
    or r0, r5
    or r0, r6
    or r0, r7
    or r0, r8
    or r0, r9
    exit
    ",
    [
        0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x06, 0x00,
        0x07, 0x00, 0x08, 0x00, 0x09
    ],
    0x9876543210
);

test_llvm!(
    test_llvm_ldxh_all2,
    "
    mov r0, r1
    ldxh r9, [r0+0]
    be16 r9
    ldxh r8, [r0+2]
    be16 r8
    ldxh r7, [r0+4]
    be16 r7
    ldxh r6, [r0+6]
    be16 r6
    ldxh r5, [r0+8]
    be16 r5
    ldxh r4, [r0+10]
    be16 r4
    ldxh r3, [r0+12]
    be16 r3
    ldxh r2, [r0+14]
    be16 r2
    ldxh r1, [r0+16]
    be16 r1
    ldxh r0, [r0+18]
    be16 r0
    or r0, r1
    or r0, r2
    or r0, r3
    or r0, r4
    or r0, r5
    or r0, r6
    or r0, r7
    or r0, r8
    or r0, r9
    exit
    ",
    [
        0x00, 0x01, 0x00, 0x02, 0x00, 0x04, 0x00, 0x08, 0x00, 0x10, 0x00, 0x20, 0x00, 0x40, 0x00,
        0x80, 0x01, 0x00, 0x02, 0x00
    ],
    0x3ff
);

test_llvm!(
    test_llvm_ldxh,
    "
    ldxh r0, [r1+2]
    exit
    ",
    [0xaa, 0xbb, 0x11, 0x22, 0xcc, 0xdd],
    0x2211
);

test_llvm!(
    test_llvm_ldxh_same_reg,
    "
    mov r0, r1
    sth [r0], 0x1234
    ldxh r0, [r0]
    exit
    ",
    [0xff, 0xff],
    0x1234
);

test_llvm!(
    test_llvm_ldxw_all,
    "
    mov r0, r1
    ldxw r9, [r0+0]
    be32 r9
    ldxw r8, [r0+4]
    be32 r8
    ldxw r7, [r0+8]
    be32 r7
    ldxw r6, [r0+12]
    be32 r6
    ldxw r5, [r0+16]
    be32 r5
    ldxw r4, [r0+20]
    be32 r4
    ldxw r3, [r0+24]
    be32 r3
    ldxw r2, [r0+28]
    be32 r2
    ldxw r1, [r0+32]
    be32 r1
    ldxw r0, [r0+36]
    be32 r0
    or r0, r1
    or r0, r2
    or r0, r3
    or r0, r4
    or r0, r5
    or r0, r6
    or r0, r7
    or r0, r8
    or r0, r9
    exit
    ",
    [
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00
    ],
    0x030f0f
);

test_llvm!(
    test_llvm_ldxw,
    "
    ldxw r0, [r1+2]
    exit
    ",
    [0xaa, 0xbb, 0x11, 0x22, 0x33, 0x44, 0xcc, 0xdd],
    0x44332211
);

const PROG_TCP_PORT_80: [u8; 152] = [
    0x71, 0x12, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x71, 0x13, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x67, 0x03, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x4f, 0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xb7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x55, 0x03, 0x0c, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x71, 0x12, 0x17, 0x00, 0x00, 0x00, 0x00, 0x00, 0x55, 0x02, 0x0a, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x71, 0x12, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x01, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
    0x57, 0x02, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x67, 0x02, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x0f, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x69, 0x12, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x15, 0x02, 0x02, 0x00, 0x00, 0x50, 0x00, 0x00, 0x69, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x55, 0x01, 0x01, 0x00, 0x00, 0x50, 0x00, 0x00, 0xb7, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x95, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

test_llvm_raw!(
    test_llvm_tcp_port80_match,
    &PROG_TCP_PORT_80,
    [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x08, 0x00, 0x45,
        0x00, 0x00, 0x56, 0x00, 0x01, 0x00, 0x00, 0x40, 0x06, 0xf9, 0x4d, 0xc0, 0xa8, 0x00, 0x01,
        0xc0, 0xa8, 0x00, 0x02, 0x27, 0x10, 0x00, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x50, 0x02, 0x20, 0x00, 0xc5, 0x18, 0x00, 0x00, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
    ],
    0x1
);

test_llvm_raw!(
    test_llvm_tcp_port80_no_match,
    &PROG_TCP_PORT_80,
    [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x08, 0x00, 0x45,
        0x00, 0x00, 0x56, 0x00, 0x01, 0x00, 0x00, 0x40, 0x06, 0xf9, 0x4d, 0xc0, 0xa8, 0x00, 0x01,
        0xc0, 0xa8, 0x00, 0x02, 0x00, 0x16, 0x27, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x51, 0x02, 0x20, 0x00, 0xc5, 0x18, 0x00, 0x00, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
    ],
    0x0
);

test_llvm_raw!(
    test_llvm_tcp_port80_no_match_ethertype,
    &PROG_TCP_PORT_80,
    [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x08, 0x01, 0x45,
        0x00, 0x00, 0x56, 0x00, 0x01, 0x00, 0x00, 0x40, 0x06, 0xf9, 0x4d, 0xc0, 0xa8, 0x00, 0x01,
        0xc0, 0xa8, 0x00, 0x02, 0x27, 0x10, 0x00, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x50, 0x02, 0x20, 0x00, 0xc5, 0x18, 0x00, 0x00, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
    ],
    0x0
);

test_llvm_raw!(
    test_llvm_tcp_port80_no_match_proto,
    &PROG_TCP_PORT_80,
    [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x08, 0x00, 0x45,
        0x00, 0x00, 0x56, 0x00, 0x01, 0x00, 0x00, 0x40, 0x11, 0xf9, 0x4d, 0xc0, 0xa8, 0x00, 0x01,
        0xc0, 0xa8, 0x00, 0x02, 0x27, 0x10, 0x00, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x50, 0x02, 0x20, 0x00, 0xc5, 0x18, 0x00, 0x00, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
    ],
    0x0
);

// TODO: boundary check, stack size
fn main() {
    test_llvm_add();
    test_llvm_alu64_arith();
    test_llvm_alu64_bit();
    test_llvm_alu_arith();
    test_llvm_alu_bit();
    test_llvm_arsh32_high_shift();
    test_llvm_arsh();
    test_llvm_arsh64();
    test_llvm_arsh_reg();

    test_llvm_be16_high();
    test_llvm_be32();
    test_llvm_be32_high();
    test_llvm_be64();

    test_llvm_div32_imm();
    test_llvm_div32_reg();
    test_llvm_div64_imm();
    test_llvm_div64_reg();

    test_llvm_div64_by_zero_imm();
    test_llvm_div_by_zero_imm();
    test_llvm_mod64_by_zero_imm();
    test_llvm_mod_by_zero_imm();
    test_llvm_div64_by_zero_reg();
    test_llvm_div_by_zero_reg();
    test_llvm_mod64_by_zero_reg();
    test_llvm_mod_by_zero_reg();

    test_llvm_exit();
    test_llvm_ja();

    test_llvm_jeq_imm();
    test_llvm_jeq_reg();
    test_llvm_jge_imm();
    test_llvm_jle_imm();
    test_llvm_jle_reg();
    test_llvm_jgt_imm();
    test_llvm_jgt_reg();
    test_llvm_jlt_imm();
    test_llvm_jlt_reg();
    test_llvm_jit_bounce();
    test_llvm_jne_reg();
    test_llvm_jset_imm();
    test_llvm_jset_reg();
    test_llvm_jsge_imm();
    test_llvm_jsge_reg();
    test_llvm_jsle_imm();
    test_llvm_jsle_reg();
    test_llvm_jsgt_imm();
    test_llvm_jsgt_reg();
    test_llvm_jslt_imm();
    test_llvm_jslt_reg();
    test_llvm_jeq32_imm();
    test_llvm_jeq32_reg();
    test_llvm_jge32_imm();
    test_llvm_jge32_reg();
    test_llvm_jgt32_imm();
    test_llvm_jgt32_reg();
    test_llvm_jle32_imm();
    test_llvm_jle32_reg();
    test_llvm_jlt32_imm();
    test_llvm_jlt32_reg();
    test_llvm_jne32_imm();
    test_llvm_jne32_reg();
    test_llvm_jset32_imm();
    test_llvm_jset32_reg();
    test_llvm_jsge32_imm();

    test_llvm_lddw();
    test_llvm_lddw2();
    test_llvm_ldxb_all();
    test_llvm_ldxb();
    test_llvm_ldxdw();
    test_llvm_ldxh_all();
    test_llvm_ldxh_all2();
    test_llvm_ldxh();
    test_llvm_ldxh_same_reg();
    test_llvm_ldxw_all();
    test_llvm_ldxw();

    test_llvm_tcp_port80_match();
    test_llvm_tcp_port80_no_match();
    test_llvm_tcp_port80_no_match_ethertype();
    test_llvm_tcp_port80_no_match_proto();
}
