use core::mem::ManuallyDrop;
use std::collections::{BTreeMap, HashMap};
use std::io::Error;
use std::path::Path;
use std::ptr;
use std::sync::atomic::AtomicUsize;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::targets::ByteOrdering;
use inkwell::types::IntType;
use inkwell::values::{BasicValue, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

use crate::ebpf::{
    self, Insn, BPF_ALU_OP_MASK, BPF_IND, BPF_JEQ, BPF_JGE, BPF_JGT, BPF_JLE, BPF_JLT, BPF_JMP32,
    BPF_JNE, BPF_JSET, BPF_JSGE, BPF_JSGT, BPF_JSLE, BPF_JSLT, BPF_X,
};

const PROG_NAME: &str = "main";

#[allow(unused)]
pub struct LLVMCompiler<'ctx> {
    context: &'ctx Context,
    module: ManuallyDrop<Module<'ctx>>,
    builder: Builder<'ctx>,
    function: FunctionValue<'ctx>,
    helpers: HashMap<u32, FunctionValue<'ctx>>,
    registers: [PointerValue<'ctx>; 11], // R0-R10
    insn_blocks: BTreeMap<u32, BasicBlock<'ctx>>,
    insn_targets: BTreeMap<u32, (BasicBlock<'ctx>, BasicBlock<'ctx>)>,
    mem_start: PointerValue<'ctx>,
    mem_end: PointerValue<'ctx>,
    umem_start: PointerValue<'ctx>,
    umem_end: PointerValue<'ctx>,
    byte_ordering: ByteOrdering,
    intrinsics: [FunctionValue<'ctx>; 3],
    entry_block: BasicBlock<'ctx>,
    stack_start_addr: IntValue<'ctx>,
    stack_end_addr: IntValue<'ctx>,
}

impl<'ctx> LLVMCompiler<'ctx> {
    pub fn new(helpers: HashMap<u32, ebpf::Helper>, context: &'ctx Context) -> Self {
        // let context = unsafe { ManuallyDrop::new() };
        let module: Module<'_> = context.create_module("ebpf_module");
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

        let helpers = helpers
            .iter()
            .map(|(&k, _)| {
                let fn_type = context.i64_type().fn_type(
                    &[
                        context.i64_type().into(),
                        context.i64_type().into(),
                        context.i64_type().into(),
                        context.i64_type().into(),
                        context.i64_type().into(),
                    ],
                    false,
                );
                let helper_function = module.add_function(&format!("helper_{}", k), fn_type, None);
                (k, helper_function)
            })
            .collect::<HashMap<_, _>>();

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

        const STACK_SIZE: usize = 512;
        let stack_array_type = context.i8_type().array_type(STACK_SIZE as u32);
        let stack_slot_ptr = builder
            .build_alloca(stack_array_type, "stack_slot")
            .unwrap();
        let int_zero = context.i64_type().const_zero(); // Index type should match pointer size potentially
        let stack_start_ptr = unsafe {
            builder
                .build_gep(
                    stack_array_type, // GEP needs the type of the value being pointed to by stack_slot_ptr
                    stack_slot_ptr,
                    &[int_zero, int_zero], // Indices: [0] selects the array itself, [0] selects the first element
                    "stack_start_ptr",
                )
                .unwrap()
        };
        let stack_start_addr = builder
            .build_ptr_to_int(stack_start_ptr, context.i64_type(), "stack_start_addr")
            .unwrap();

        let stack_size_val = context.i64_type().const_int(STACK_SIZE as _, false);
        let stack_end_addr = builder
            .build_int_add(
                stack_start_addr,
                stack_size_val,
                "stack_end_addr", // This corresponds to cranelift's stack_addr(..., ss, STACK_SIZE)
            )
            .unwrap();

        // Initialize other registers to 0
        let zero = context.i64_type().const_int(0, false);
        for i in 0..=10 {
            builder
                .build_store(registers[i].unwrap(), zero)
                .unwrap()
                .set_alignment(8)
                .unwrap();
        }
        builder
            .build_store(registers[1].unwrap(), mem_start)
            .unwrap()
            .set_alignment(8)
            .unwrap();
        builder
            .build_store(registers[2].unwrap(), mem_end)
            .unwrap()
            .set_alignment(8)
            .unwrap();
        builder
            .build_store(registers[10].unwrap(), stack_end_addr)
            .unwrap()
            .set_alignment(8)
            .unwrap();

        LLVMCompiler {
            context,
            module: ManuallyDrop::new(module),
            builder,
            helpers,
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
            stack_start_addr,
            stack_end_addr,
        }
    }

    pub fn compile_function(&mut self, prog: &[u8]) -> Result<Vec<u8>, Error> {
        build_cfg(
            self.context,
            &mut self.insn_blocks,
            &mut self.insn_targets,
            prog,
            self.function,
            AtomicUsize::new(0),
        )?;

        self.translate_program(prog)
    }

    fn translate_program(&mut self, prog: &[u8]) -> Result<Vec<u8>, Error> {
        let ctx = self.context;
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
                        ebpf::LD_ABS_B | ebpf::LD_IND_B => ctx.i8_type(),
                        ebpf::LD_ABS_H | ebpf::LD_IND_H => ctx.i16_type(),
                        ebpf::LD_ABS_W | ebpf::LD_IND_W => ctx.i32_type(),
                        ebpf::LD_ABS_DW | ebpf::LD_IND_DW => ctx.i64_type(),
                        _ => unreachable!(),
                    };

                    let mem_start = self.mem_start.as_basic_value_enum().into_int_value();
                    let offset = ctx.i64_type().const_int(insn.off as u64, false);
                    let addr = builder.build_int_add(mem_start, offset, "addr").unwrap();

                    // IND instructions additionally add the value of the source register
                    let is_ind = (insn.opc & BPF_IND) != 0;
                    let addr = if is_ind {
                        let src_reg = self.insn_src(&insn);
                        builder
                            .build_int_add(addr, src_reg, "ind_addr_with)_src")
                            .unwrap()
                    } else {
                        addr
                    };

                    let addr =
                        addr.const_to_pointer(self.context.ptr_type(AddressSpace::default()));
                    let loaded = self
                        .builder
                        .build_load(ty, addr, "loaded")
                        .unwrap()
                        .into_int_value();
                    let ext = if ty != ctx.i64_type() {
                        builder
                            .build_int_z_extend(loaded, ctx.i64_type(), "ext")
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
                    let iconst = ctx.i64_type().const_int(imm as u64, true);
                    self.set_dst(&insn, iconst);
                }
                // BPF_LDX class
                ebpf::LD_B_REG | ebpf::LD_H_REG | ebpf::LD_W_REG | ebpf::LD_DW_REG => {
                    let ty = match insn.opc {
                        ebpf::LD_B_REG => ctx.i8_type(),
                        ebpf::LD_H_REG => ctx.i16_type(),
                        ebpf::LD_W_REG => ctx.i32_type(),
                        ebpf::LD_DW_REG => ctx.i64_type(),
                        _ => unreachable!(),
                    };

                    let base = self.insn_src(&insn);
                    let offset = ty.const_int(insn.off as u64, false);
                    let addr = self
                        .builder
                        .build_int_add(base, offset, "addr_with_addr")
                        .unwrap();
                    let addr =
                        addr.const_to_pointer(self.context.ptr_type(AddressSpace::default()));
                    let loaded = self
                        .builder
                        .build_load(ty, addr, "loaded")
                        .unwrap()
                        .into_int_value();
                    // let loaded = self.reg_load(ty, base, insn.off);

                    let ext = if ty != ctx.i64_type() {
                        builder
                            .build_int_z_extend(loaded, ctx.i64_type(), "ext")
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
                        ebpf::ST_B_IMM | ebpf::ST_B_REG => ctx.i8_type(),
                        ebpf::ST_H_IMM | ebpf::ST_H_REG => ctx.i16_type(),
                        ebpf::ST_W_IMM | ebpf::ST_W_REG => ctx.i32_type(),
                        ebpf::ST_DW_IMM | ebpf::ST_DW_REG => ctx.i64_type(),
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

                    let narrow = if ty != ctx.i64_type() {
                        builder.build_int_truncate(value, ty, "narrow").unwrap()
                    } else {
                        value
                    };

                    let base = self.insn_dst(&insn);
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
                        ctx.i32_type().const_int(0, false)
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
                    let zero = ctx.i32_type().const_int(0, false);
                    let one = ctx.i32_type().const_int(1, false);

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
                        .build_select(rhs_is_zero, zero, div_res, "res")
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
                    let zero = ctx.i32_type().const_int(0, false);
                    let one = ctx.i32_type().const_int(1, false);

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
                        ctx.i64_type().const_int(0, false)
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
                    let zero = ctx.i64_type().const_int(0, false);
                    let one = ctx.i64_type().const_int(1, false);

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
                        .build_select(rhs_is_zero, zero, div_res, "res")
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
                    let zero = ctx.i64_type().const_int(0, false);
                    let one = ctx.i64_type().const_int(1, false);

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
                        .build_load(ctx.i64_type(), self.registers[0], "ret_val")
                        .unwrap();
                    r0.as_instruction_value().unwrap().set_alignment(8).unwrap();
                    self.builder.build_return(Some(&r0)).unwrap();
                    prev_is_terminator = true;
                }
                ebpf::TAIL_CALL => unimplemented!(),
                ebpf::CALL => {
                    let k = insn.imm as u32;
                    let func = self.helpers.get(&k).copied().ok_or_else(|| {
                        Error::new(
                            std::io::ErrorKind::Other,
                            format!("[LLVM] Error: unknown helper function (id: {:#x})", k),
                        )
                    })?;

                    let arg = |i: usize| {
                        let inst = self
                            .builder
                            .build_load(ctx.i64_type(), self.registers[i], &format!("arg{}", i))
                            .unwrap();
                        inst.as_instruction_value()
                            .unwrap()
                            .set_alignment(8)
                            .unwrap();
                        inst
                    };

                    let args = [
                        arg(1).into(),
                        arg(2).into(),
                        arg(3).into(),
                        arg(4).into(),
                        arg(5).into(),
                    ];

                    let res = self
                        .builder
                        .build_call(func, &args, &format!("call_{}", k))
                        .unwrap();

                    // store to R0 according to eBPF ABI
                    // TODO: does llvm provide an interface for the reg to store the return value?
                    let inst = self
                        .builder
                        .build_store(
                            self.registers[0],
                            res.try_as_basic_value().left().unwrap().into_int_value(),
                        )
                        .unwrap();
                    inst.set_alignment(8).unwrap();
                }
                _ => unimplemented!("inst: {:?}", insn),
            }
            insn_ptr += 1;
        }
        self.module.write_bitcode_to_path(Path::new("/tmp/a.bc"));
        let bc = self.module.write_bitcode_to_memory().as_slice().to_vec();
        {
            let membuf = MemoryBuffer::create_from_memory_range(&bc, "t");
            let m = Module::parse_bitcode_from_buffer(&membuf, ctx).unwrap();
        }
        Ok(bc)
    }

    fn reg_load<'b>(&'b self, ty: IntType<'b>, base: IntValue<'b>, offset: i16) -> IntValue<'b>
    where
        'b: 'ctx,
    {
        // self.insert_bounds_check(bcx, ty, base, offset);
        // bcx.ins().load(ty, MemFlags::new(), base, offset as i32)
        if offset == 0 {
            return self
                .builder
                .build_load(
                    ty,
                    base.const_to_pointer(self.context.ptr_type(AddressSpace::default())),
                    "loaded",
                )
                .unwrap()
                .into_int_value();
        } else {
            let offset = ty.const_int(offset as u64, false);
            let addr = self
                .builder
                .build_int_add(base, offset, "addr_with_addr")
                .unwrap();
            let addr = addr.const_to_pointer(self.context.ptr_type(AddressSpace::default()));
            self.builder
                .build_load(ty, addr, "loaded")
                .unwrap()
                .into_int_value()
        }
    }

    // TODO: signed or unsigned extend?
    fn reg_store(&self, ty: IntType<'_>, base: IntValue<'_>, offset: i16, val: IntValue<'_>) {
        if offset == 0 {
            self.builder
                .build_store(
                    base.const_to_pointer(self.context.ptr_type(AddressSpace::default())),
                    val,
                )
                .unwrap();
        } else {
            let offset = ty.const_int(offset as u64, false);
            let addr = self.builder.build_int_add(base, offset, "addr").unwrap();
            let inst = self
                .builder
                .build_store(
                    addr.const_to_pointer(self.context.ptr_type(AddressSpace::default())),
                    val,
                )
                .unwrap();
            inst.set_alignment(8).unwrap();
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
        let load = self
            .builder
            .build_load(self.context.i64_type(), dst, "dst_val")
            .unwrap();
        load.as_instruction_value()
            .unwrap()
            .set_alignment(8)
            .unwrap();
        let dst_val = load.into_int_value();
        dst_val
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
        let inst = self
            .builder
            .build_load(self.context.i64_type(), src, "src_val")
            .unwrap();
        inst.as_instruction_value()
            .unwrap()
            .set_alignment(8)
            .unwrap();
        inst.into_int_value()
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
        let inst = self.builder.build_store(dst, val).unwrap();
        inst.set_alignment(8).unwrap();
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

    #[allow(unused)]
    pub fn print_ir(&self) {
        self.module.print_to_stderr();
    }

    pub fn execute(
        &self,
        helpers: &HashMap<u32, ebpf::Helper>,
        mem_ptr: *const u8,
        mem_len: u64,
        _mbuff_ptr: *const u8,
        _mbuff_len: u64,
    ) -> u64 {
        let ee = self
            .module
            .create_jit_execution_engine(inkwell::OptimizationLevel::None)
            .unwrap();
        for (k, &v) in helpers.iter() {
            let func = self.helpers.get(k).unwrap();
            let func_addr: usize = unsafe { std::mem::transmute(v) };
            ee.add_global_mapping(func, func_addr);
        }
        let func = ee.get_function_address("main").unwrap();
        let func: extern "C" fn(*const u8, u64, *const u8, u64) -> u64 =
            unsafe { std::mem::transmute(func) };
        func(mem_ptr, mem_len, ptr::null(), 0)
    }
}

fn gen_next_label(label_cnt: &AtomicUsize) -> String {
    let cnt = label_cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!("block_{}", cnt)
}

/// Analyze the program and build the CFG
///
/// We do this because cranelift does not allow us to switch back to a previously
/// filled block and add instructions to it. So we can't split the program as we
/// translate it.
fn build_cfg<'ctx>(
    ctx: &'ctx Context,
    insn_blocks: &mut BTreeMap<u32, BasicBlock<'ctx>>,
    insn_targets: &mut BTreeMap<u32, (BasicBlock<'ctx>, BasicBlock<'ctx>)>,
    prog: &[u8],
    function: FunctionValue<'ctx>,
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
                    ctx,
                    insn_blocks,
                    insn_targets,
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

fn prepare_jump_blocks<'ctx>(
    ctx: &'ctx Context,
    insn_blocks: &mut BTreeMap<u32, BasicBlock<'ctx>>,
    insn_targets: &mut BTreeMap<u32, (BasicBlock<'ctx>, BasicBlock<'ctx>)>,
    insn_ptr: usize,
    insn: &Insn,
    function: FunctionValue<'ctx>,
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

#[test]
fn t1() {
    let ctx = Context::create();
    let bc = std::fs::read("/tmp/a.bc").unwrap();
    let membuf = MemoryBuffer::create_from_memory_range(&bc, "t");
    let m = Module::parse_bitcode_from_buffer(&membuf, &ctx).unwrap();
}
