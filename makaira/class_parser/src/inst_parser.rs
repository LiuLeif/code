// 2019-09-03 13:53
use nom::bytes::complete::*;
use nom::character::complete::*;
use nom::number::complete::*;
// use nom::bits::complete::*;

use nom::branch::*;
use nom::combinator::*;
use nom::error::*;
use nom::multi::*;
use nom::sequence::*;
use nom::*;

use std::cell::Cell;

#[derive(Debug)]
pub enum MakairaInst {
    AALOAD,
    AASTORE,
    ACONST_NULL,
    ALOAD(u8),
    ALOAD_0,
    ALOAD_1,
    ALOAD_2,
    ALOAD_3,
    ANEWARRAY(u16),
    ARETURN,
    ARRAYLENGTH,
    ASTORE(u8),
    ASTORE_0,
    ASTORE_1,
    ASTORE_2,
    ASTORE_3,
    ATHROW,
    BALOAD,
    BASTORE,
    BIPUSH(u8),
    BREAKPOINT,
    CALOAD,
    CASTORE,
    CHECKCAST(u16),
    D2F,
    D2I,
    D2L,
    DADD,
    DALOAD,
    DASTORE,
    DCMPG,
    DCMPL,
    DCONST_0,
    DCONST_1,
    DDIV,
    DLOAD(u8),
    DLOAD_0,
    DLOAD_1,
    DLOAD_2,
    DLOAD_3,
    DMUL,
    DNEG,
    DREM,
    DRETURN,
    DSTORE(u8),
    DSTORE_0,
    DSTORE_1,
    DSTORE_2,
    DSTORE_3,
    DSUB,
    DUP,
    DUP_X1,
    DUP_X2,
    DUP2,
    DUP2_X1,
    DUP2_X2,
    F2D,
    F2I,
    F2L,
    FADD,
    FALOAD,
    FASTORE,
    FCMPG,
    FCMPL,
    FCONST_0,
    FCONST_1,
    FCONST_2,
    FDIV,
    FLOAD(u8),
    FLOAD_0,
    FLOAD_1,
    FLOAD_2,
    FLOAD_3,
    FMUL,
    FNEG,
    FREM,
    FRETURN,
    FSTORE(u8),
    FSTORE_0,
    FSTORE_1,
    FSTORE_2,
    FSTORE_3,
    FSUB,
    GETFIELD(u16),
    GETSTATIC(u16),
    GOTO(u16),
    GOTO_W(u32),
    I2B,
    I2C,
    I2D,
    I2F,
    I2L,
    I2S,
    IADD,
    IALOAD,
    IAND,
    IASTORE,
    ICONST_M1,
    ICONST_0,
    ICONST_1,
    ICONST_2,
    ICONST_3,
    ICONST_4,
    ICONST_5,
    IDIV,
    IF_ACMPEQ(u16),
    IF_ACMPNE(u16),
    IF_ICMPEQ(u16),
    IF_ICMPGE(u16),
    IF_ICMPGT(u16),
    IF_ICMPLE(u16),
    IF_ICMPLT(u16),
    IF_ICMPNE(u16),
    IFEQ(u16),
    IFGE(u16),
    IFGT(u16),
    IFLE(u16),
    IFLT(u16),
    IFNE(u16),
    IFNONNULL(u16),
    IFNULL(u16),
    IINC(u16),
    ILOAD(u8),
    ILOAD_0,
    ILOAD_1,
    ILOAD_2,
    ILOAD_3,
    IMPDEP1,
    IMPDEP2,
    IMUL,
    INEG,
    INSTANCEOF(u16),
    INVOKEINTERFACE(u16, u8),
    INVOKESPECIAL(u16),
    INVOKESTATIC(u16),
    INVOKEVIRTUAL(u16),
    IOR,
    IREM,
    IRETURN,
    ISHL,
    ISHR,
    ISTORE(u8),
    ISTORE_0,
    ISTORE_1,
    ISTORE_2,
    ISTORE_3,
    ISUB,
    IUSHR,
    IXOR,
    JSR(u16),
    JSR_W(u32),
    L2D,
    L2F,
    L2I,
    LADD,
    LALOAD,
    LAND,
    LASTORE,
    LCMP,
    LCONST_0,
    LCONST_1,
    LDC(u8),
    LDC_W(u16),
    LDC2_W(u16),
    LDIV,
    LLOAD(u8),
    LLOAD_0,
    LLOAD_1,
    LLOAD_2,
    LLOAD_3,
    LMUL,
    LNEG,
    LOOKUPSWITCH(u32, Vec<(u32, u32)>),
    LOR,
    LREM,
    LRETURN,
    LSHL,
    LSHR,
    LSTORE(u8),
    LSTORE_0,
    LSTORE_1,
    LSTORE_2,
    LSTORE_3,
    LSUB,
    LUSHR,
    LXOR,
    MONITORENTER,
    MONITOREXIT,
    MULTIANEWARRAY(u16, u8),
    NEW(u16),
    NEWARRAY(u8),
    NOP,
    POP,
    POP2,
    PUTFIELD(u16),
    PUTSTATIC(u16),
    RET(u8),
    RETURN,
    SALOAD,
    SASTORE,
    SIPUSH(u16),
    SWAP,
    TABLESWITCH,
    WIDE,
}

fn parse_inst(input: &[u8]) -> IResult<&[u8], MakairaInst> {
    use MakairaInst::*;

    let mut input = input;
    let (tmp, opcode) = be_u8(input)?;
    input = tmp;
    let inst = {
        match opcode {
            0x32 => AALOAD,
            0x53 => AASTORE,
            0x01 => ACONST_NULL,
            0x2a => ALOAD_0,
            0x2b => ALOAD_1,
            0x2c => ALOAD_2,
            0x2d => ALOAD_3,
            0xb0 => ARETURN,
            0xbe => ARRAYLENGTH,
            0x4b => ASTORE_0,
            0x4c => ASTORE_1,
            0x4d => ASTORE_2,
            0x4e => ASTORE_3,
            0xbf => ATHROW,
            0x33 => BALOAD,
            0x54 => BASTORE,
            0x34 => CALOAD,
            0x55 => CASTORE,
            0x90 => D2F,
            0x8e => D2I,
            0x8f => D2L,
            0x63 => DADD,
            0x31 => DALOAD,
            0x52 => DASTORE,
            0x98 => DCMPG,
            0x97 => DCMPL,
            0x0e => DCONST_0,
            0x0f => DCONST_1,
            0x6f => DDIV,
            0x26 => DLOAD_0,
            0x27 => DLOAD_1,
            0x28 => DLOAD_2,
            0x29 => DLOAD_3,
            0x6b => DMUL,
            0x77 => DNEG,
            0x73 => DREM,
            0xaf => DRETURN,
            0x47 => DSTORE_0,
            0x48 => DSTORE_1,
            0x49 => DSTORE_2,
            0x50 => DSTORE_3,
            0x67 => DSUB,
            0x59 => DUP,
            0x5a => DUP_X1,
            0x5b => DUP_X2,
            0x5c => DUP2,
            0x5d => DUP2_X1,
            0x5e => DUP2_X2,
            0x8d => F2D,
            0x8b => F2I,
            0x8c => F2L,
            0x62 => FADD,
            0x30 => FALOAD,
            0x51 => FASTORE,
            0x96 => FCMPG,
            0x95 => FCMPL,
            0x0b => FCONST_0,
            0x0c => FCONST_1,
            0x0d => FCONST_2,
            0x6e => FDIV,
            0x22 => FLOAD_0,
            0x23 => FLOAD_1,
            0x24 => FLOAD_2,
            0x25 => FLOAD_3,
            0x6a => FMUL,
            0x76 => FNEG,
            0x72 => FREM,
            0xae => FRETURN,
            0x43 => FSTORE_0,
            0x44 => FSTORE_1,
            0x45 => FSTORE_2,
            0x46 => FSTORE_3,
            0x66 => FSUB,
            0x91 => I2B,
            0x92 => I2C,
            0x87 => I2D,
            0x86 => I2F,
            0x85 => I2L,
            0x93 => I2S,
            0x60 => IADD,
            0x2e => IALOAD,
            0x7e => IAND,
            0x4f => IASTORE,
            0x02 => ICONST_M1,
            0x03 => ICONST_0,
            0x04 => ICONST_1,
            0x05 => ICONST_2,
            0x06 => ICONST_3,
            0x07 => ICONST_4,
            0x08 => ICONST_5,
            0x6c => IDIV,
            0x1a => ILOAD_0,
            0x1b => ILOAD_1,
            0x1c => ILOAD_2,
            0x1d => ILOAD_3,
            0xfe => IMPDEP1,
            0xff => IMPDEP2,
            0x68 => IMUL,
            0x74 => INEG,
            0x80 => IOR,
            0x70 => IREM,
            0xac => IRETURN,
            0x78 => ISHL,
            0x7a => ISHR,
            0x3b => ISTORE_0,
            0x3c => ISTORE_1,
            0x3d => ISTORE_2,
            0x3e => ISTORE_3,
            0x64 => ISUB,
            0x7c => IUSHR,
            0x82 => IXOR,
            0x8a => L2D,
            0x89 => L2F,
            0x88 => L2I,
            0x61 => LADD,
            0x2f => LALOAD,
            0x7f => LAND,
            0x50 => LASTORE,
            0x94 => LCMP,
            0x09 => LCONST_0,
            0x0a => LCONST_1,
            0x6d => LDIV,
            0x1e => LLOAD_0,
            0x1f => LLOAD_1,
            0x20 => LLOAD_2,
            0x21 => LLOAD_3,
            0x69 => LMUL,
            0x75 => LNEG,
            0x81 => LOR,
            0x71 => LREM,
            0xad => LRETURN,
            0x79 => LSHL,
            0x7b => LSHR,
            0x3f => LSTORE_0,
            0x40 => LSTORE_1,
            0x41 => LSTORE_2,
            0x42 => LSTORE_3,
            0x65 => LSUB,
            0x7d => LUSHR,
            0x83 => LXOR,
            0xc2 => MONITORENTER,
            0xc3 => MONITOREXIT,
            0x00 => NOP,
            0x57 => POP,
            0x58 => POP2,
            0xb1 => RETURN,
            0x35 => SALOAD,
            0x56 => SASTORE,
            0x5f => SWAP,
            0x19 | 0x3a | 0x10 | 0x18 | 0x39 | 0x17 | 0x38 | 0x15 | 0x36 | 0x12 | 0x16 | 0x37
            | 0xbc | 0xa9 => {
                let (tmp, index) = be_u8(input)?;
                input = tmp;
                match opcode {
                    0x19 => ALOAD(index),
                    0x3a => ASTORE(index),
                    0x10 => BIPUSH(index),
                    0x18 => DLOAD(index),
                    0x39 => DSTORE(index),
                    0x17 => FLOAD(index),
                    0x38 => FSTORE(index),
                    0x15 => ILOAD(index),
                    0x36 => ISTORE(index),
                    0x12 => LDC(index),
                    0x16 => LLOAD(index),
                    0x37 => LSTORE(index),
                    0xbc => NEWARRAY(index),
                    0xa9 => RET(index),
                    _ => panic!(),
                }
            }
            0x64 | 0xb2 | 0xa7 | 0xb7 | 0xbd | 0xc0 | 0xa5 | 0xa6 | 0x9f | 0xa2 | 0xa3 | 0xa4
            | 0xa1 | 0xa0 | 0x99 | 0x9c | 0x9d | 0x9e | 0x9b | 0x9a | 0xc7 | 0xc6 | 0x84 | 0xc1
            | 0xb7 | 0xb8 | 0xb6 | 0xa8 | 0x13 | 0x14 | 0xbb | 0xb5 | 0xb3 | 0x11 => {
                let (tmp, value) = be_u16(input)?;
                input = tmp;
                match opcode {
                    0xbd => ANEWARRAY(value),
                    0xc0 => CHECKCAST(value),
                    0x64 => GETFIELD(value),
                    0xb2 => GETSTATIC(value),
                    0xa7 => GOTO(value),
                    0xa5 => IF_ACMPEQ(value),
                    0xa6 => IF_ACMPNE(value),
                    0x9f => IF_ICMPEQ(value),
                    0xa2 => IF_ICMPGE(value),
                    0xa3 => IF_ICMPGT(value),
                    0xa4 => IF_ICMPLE(value),
                    0xa1 => IF_ICMPLT(value),
                    0xa0 => IF_ICMPNE(value),
                    0x99 => IFEQ(value),
                    0x9c => IFGE(value),
                    0x9d => IFGT(value),
                    0x9e => IFLE(value),
                    0x9b => IFLT(value),
                    0x9a => IFNE(value),
                    0xc7 => IFNONNULL(value),
                    0xc6 => IFNULL(value),
                    0x84 => IINC(value),
                    0xc1 => INSTANCEOF(value),
                    0xb7 => INVOKESPECIAL(value),
                    0xb8 => INVOKESTATIC(value),
                    0xb6 => INVOKEVIRTUAL(value),
                    0xa8 => JSR(value),
                    0x13 => LDC_W(value),
                    0x14 => LDC2_W(value),
                    0xbb => NEW(value),
                    0xb5 => PUTFIELD(value),
                    0xb3 => PUTSTATIC(value),
                    0x11 => SIPUSH(value),
                    _ => panic!(),
                }
            }

            0xc5 => {
                let (tmp, (value, dimen)) = pair(be_u16, be_u8)(input)?;
                input = tmp;
                MULTIANEWARRAY(value, dimen)
            }

            0xc8 | 0xa8 => {
                let (tmp, value) = be_u32(input)?;
                input = tmp;
                match opcode {
                    0xc8 => GOTO_W(value),
                    0xa8 => JSR_W(value),
                    _ => panic!(),
                }
            }

            0xb9 => {
                let (tmp, (value, count, _)) = tuple((be_u16, be_u8, be_u8))(input)?;
                INVOKEINTERFACE(value, count)
            }

            0xab => {
                let total_len = INST_LEN.with(|len| len.get());
                let padding = (4 - (total_len - input.len() as i32) % 4) % 4;
                let (tmp, (default_value, n_pairs)) = pair(be_u32, be_u32)(input)?;
                let (tmp, pairs) = count(pair(be_u32, be_u32), n_pairs as usize)(tmp)?;
                input = tmp;
                LOOKUPSWITCH(default_value, pairs)
            }
            // _ => Err(Err::Error((input, ErrorKind::Tag))),
            _ => panic!("unknown inst"),
        }
    };
    Ok((input, inst))
}

thread_local! {
    static INST_LEN: Cell<i32> = Cell::new(0);
}

pub fn parse(input: &[u8]) -> IResult<&[u8], Vec<MakairaInst>> {
    INST_LEN.with(|len| len.set(input.len() as i32));
    many0(parse_inst)(input)
}
