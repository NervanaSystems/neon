package MaxAs::MaxAsGrammar;

use strict;
use Carp;
use Exporter;
use Data::Dumper;
our @ISA = qw(Exporter);

our @EXPORT = qw(
    %grammar %flags
    parseInstruct genCode genReuseCode
    processAsmLine processSassLine processSassCtrlLine
    replaceXMADs printCtrl readCtrl getRegNum getVecRegisters getAddrVecRegisters
);

require 5.10.0;

# Helper functions for operands
sub getI
{
    my ($orig, $pos, $mask) = @_;
    my $val = $orig;
    my $neg = $val =~ s|^\-||;

    # parse out our custom index immediates for addresses
    if ($val  =~ m'^(\d+)[xX]<([^>]+)>')
    {
        # allow any perl expression and multiply result by leading decimal.
        # also allow global scalar varibles in the expression.
        my $mul = $1;
        my $exp = $2;
        # strip leading zeros (don't interpret numbers as octal)
        $exp =~ s/(?<!\d)0+(?=[1-9])//g;
        my @globals = $exp =~ m'\$\w+'g;
        my $our = @globals ? ' our (' . join(',',@globals) . ');' : '';
        $val = $mul * eval "package MaxAs::MaxAs::CODE;$our $exp";
        #print "$val = $mul x $exp\n"; # if $our;
    }
    # hexidecial value
    elsif ($val  =~ m'^0x[0-9a-zA-Z]+')
    {
        $val = hex($val);
    }
    # otherwise val is a simple decimal value that doesn't need to be modified

    if ( $neg )
    {
        # if the mask removes the sign bit the "neg" flag adds it back on the code somewhere else
        $val = -$val;
        $val &= $mask;
    }
    if (($val & $mask) != $val)
    {
        die sprintf "Immediate value out of range(0x%x): 0x%x ($orig)\n", $mask, $val;
    }
    return $val << $pos;
}
sub getF
{
    my ($val, $pos, $type, $trunc) = @_;
    # hexidecial value
    if ($val  =~ m'^0x[0-9a-zA-Z]+')
    {
        $val = hex($val);
    }
    # support infinity
    elsif ($val =~ m'INF'i)
    {
        $val = $trunc ? ($type eq 'f' ? 0x7f800 : 0x7ff00) : 0x7f800000;
    }
    else
    {
        $val = unpack(($type eq 'f' ? 'L' : 'Q'), pack $type, $val);

        # strip off sign bit if truncating.  It will added elsewhere in the code by the flag capture.
        $val = ($val >> $trunc) & 0x7ffff if $trunc;
    }
    return $val << $pos;
}
sub getR
{
    my ($val, $pos) = @_;
    if ($val =~ m'^R(\d+|Z)$' && $1 < 255)
    {
        $val = $1 eq 'Z' ? 0xff : $1;
    }
    else
    {
        die "Bad register name found: $val\n";
    }
    return $val << $pos;
}
sub getP
{
    my ($val, $pos) = @_;
    if ($val =~ m'^P(\d|T)$' && $1 < 7)
    {
        $val = $1 eq 'T' ? 7 : $1;
    }
    else
    {
        die "Bad predicate name found: $val\n";
    }
    return $val << $pos;
}
sub getC { ((hex($_[0]) >> 2) & 0x7fff) << 20 }

# Map operands into their value and position in the op code.
my %operands =
(
    p0      => sub { getP($_[0], 0)  },
    p3      => sub { getP($_[0], 3)  },
    p12     => sub { getP($_[0], 12) },
    p29     => sub { getP($_[0], 29) },
    p39     => sub { getP($_[0], 39) },
    p45     => sub { getP($_[0], 45) },
    p48     => sub { getP($_[0], 48) },
    p58     => sub { getP($_[0], 58) },
    r0      => sub { getR($_[0], 0)  },
    r8      => sub { getR($_[0], 8)  },
    r20     => sub { getR($_[0], 20) },
    r28     => sub { getR($_[0], 28) },
    r39s20  => sub { getR($_[0], 39) },
    r39     => sub { getR($_[0], 39) },
    r39a    => sub { getR($_[0], 39) }, # does not modify op code, xor the r39 value again to whipe it out, register must be in sequence with r20
    c20     => sub { getC($_[0])     },
    c39     => sub { getC($_[0])     },
    c34     => sub { hex($_[0]) << 34 },
    c36     => sub { hex($_[0]) << 36 },
    f20w32  => sub { getF($_[0], 20, 'f')        },
    f20     => sub { getF($_[0], 20, 'f', 12)    },
    d20     => sub { getF($_[0], 20, 'd', 44)    },
    i8w4    => sub { getI($_[0], 8,  0xf)        },
    i20     => sub { getI($_[0], 20, 0x7ffff)    },
    i20w6   => sub { getI($_[0], 20, 0x3f)       },
    i20w7   => sub { getI($_[0], 20, 0x7f)       },
    i20w8   => sub { getI($_[0], 20, 0xff)       },
    i20w12  => sub { getI($_[0], 20, 0xfff)      },
    i20w24  => sub { getI($_[0], 20, 0xffffff)   },
    i20w32  => sub { getI($_[0], 20, 0xffffffff) },
    i31w4   => sub { getI($_[0], 31, 0xf)        },
    i34w13  => sub { getI($_[0], 34, 0x1fff)     },
    i36w20  => sub { getI($_[0], 36, 0xfffff)    },
    i39w8   => sub { getI($_[0], 39, 0xff)       },
    i28w8   => sub { getI($_[0], 28, 0xff)       },
    i28w20  => sub { getI($_[0], 28, 0xfffff)    },
    i48w8   => sub { getI($_[0], 48, 0xff)       },
    i51w5   => sub { getI($_[0], 51, 0x1f)       },
    i53w5   => sub { getI($_[0], 53, 0x1f)       },
);

# Rules for operands and their closely tied flags
my $hex     = qr"0[xX][0-9a-fA-F]+";
my $iAddr   = qr"\d+[xX]<[^>]+>";
my $immed   = qr"$hex|$iAddr|\d+"o;
my $reg     = qr"[a-zA-Z_]\w*"; # must start with letter or underscore\
my $p       = qr"P[0-6T]";
my $noPred  = qr"(?<noPred>)";
my $pred    = qr"\@(?<predNot>\!)?P(?<predNum>[0-6]) ";
my $p0      = qr"(?<p0>$p)"o;
my $p3      = qr"(?<p3>$p)"o;
my $p12     = qr"(?<p12not>\!)?(?<p12>$p)"o;
my $p29     = qr"(?<p29not>\!)?(?<p29>$p)"o;
my $p39     = qr"(?<p39not>\!)?(?<p39>$p)"o;
my $p45     = qr"(?<p45>$p)"o;
my $p48     = qr"(?<p48>$p)"o;
my $p58     = qr"(?<p58>$p)"o;
my $r0      = qr"(?<r0>$reg)";
my $r0cc    = qr"(?<r0>$reg)(?<CC>\.CC)?";
my $r8      = qr"(?<r8neg>\-)?(?<r8abs>\|)?(?<r8>$reg)\|?(?:\.(?<r8part>H0|H1|B0|B1|B2|B3|H0_H0|H1_H1))?(?<reuse1>\.reuse)?";
my $r20     = qr"(?<r20neg>\-)?(?<r20abs>\|)?(?<r20>$reg)\|?(?:\.(?<r20part>H0|H1|B0|B1|B2|B3|H0_H0|H1_H1))?(?<reuse2>\.reuse)?";
my $r28     = qr"(?<r28>$reg)";
my $r39s20  = qr"(?<r20neg>\-)?(?<r20abs>\|)?(?<r39s20>(?<r20>$reg))\|?(?:\.(?<r39part>H0|H1))?(?<reuse2>\.reuse)?";
my $r39     = qr"(?<r39neg>\-)?(?<r39>$reg)(?:\.(?<r39part>H0|H1))?(?<reuse3>\.reuse)?";
my $r39a    = qr"(?<r39a>(?<r39>$reg))(?<reuse3>\.reuse)?";
my $c20     = qr"(?<r20neg>\-)?(?<r20abs>\|)?c\[(?<c34>$hex)\]\s*\[(?<c20>$hex)\]\|?(?:\.(?<r20part>H0|H1|B0|B1|B2|B3))?"o;
my $c20x    = qr"(?<r20neg>\-)?(?<r20abs>\|)?c\[(?<c34>$hex)\]\s*\[(?<c20>$hex)\]\|?(?:\.(?<r20partx>H0|H1|B0|B1|B2|B3))?"o;
my $c20s39  = qr"(?<r39neg>\-)?c\[(?<c34>$hex)\]\s*\[(?<c39>$hex)\]"o;
my $f20w32  = qr"(?<f20w32>(?:\-|\+|)(?i:$hex|inf\s*|\d+(?:\.\d+(?:e[\+\-]\d+)?)?))";
my $f20     = qr"(?<f20>(?:(?<neg>\-)|\+|)(?i:inf\s*|\d+(?:\.\d+(?:e[\+\-]\d+)?)?))(?<r20neg>\.NEG)?"o;
my $d20     = qr"(?<d20>(?:(?<neg>\-)|\+|)(?i:inf\s*|\d+(?:\.\d+(?:e[\+\-]\d+)?)?))(?<r20neg>\.NEG)?"o;
my $i8w4    = qr"(?<i8w4>$immed)"o;
my $i20     = qr"(?<i20>(?<neg>\-)?$immed)(?<r20neg>\.NEG)?"o;
my $i20w6   = qr"(?<i20w6>$immed)"o;
my $i20w7   = qr"(?<i20w7>$immed)"o;
my $i20w8   = qr"(?<i20w8>$immed)"o;
my $i20w12  = qr"(?<i20w12>$immed)"o;
my $i20w24  = qr"(?<i20w24>\-?$immed)"o;
my $i20w32  = qr"(?<i20w32>\-?$immed)"o;
my $i39w8   = qr"(?<i39w8>\-?$immed)"o;
my $i28w8   = qr"(?<i28w8>$immed)"o;
my $i28w20  = qr"(?<i28w20>\-?$immed)"o;
my $i31w4   = qr"(?<i31w4>$immed)"o;
my $i34w13  = qr"(?<i34w13>$immed)"o;
my $i36w20  = qr"(?<i36w20>$immed)"o;
my $i48w8   = qr"(?<i48w8>$immed)"o;
my $i51w5   = qr"(?<i51w5>$immed)"o;
my $i53w5   = qr"(?<i53w5>$immed)"o;
my $ir20    = qr"$i20|$r20"o;
my $cr20    = qr"$c20|$r20"o;
my $icr20   = qr"$i20|$c20|$r20"o;
my $fcr20   = qr"$f20|$c20|$r20"o;
my $cr39    = qr"$c20s39|$r39"o;
my $dr20    = qr"$d20|$r20"o;

# Instruction specific rules for capturing various flags
my $u32   = qr"(?<U32>\.U32)?";
my $ftz   = qr"(?<FTZ>\.FTZ)?";
my $sat   = qr"(?<SAT>\.SAT)?";
my $rnd   = qr"(?:\.(?<rnd>RN|RM|RP|RZ))?";
my $round = qr"(?:\.(?<round>ROUND|FLOOR|CEIL|TRUNC))?";
my $fcmp  = qr"(?<cmp>\.LT|\.EQ|\.LE|\.GT|\.NE|\.GE|\.NUM|\.NAN|\.LTU|\.EQU|\.LEU|\.GTU|\.NEU|\.GEU|)";
my $icmp  = qr"\.(?<cmp>LT|EQ|LE|GT|NE|GE)";
my $bool  = qr"\.(?<bool>AND|OR|XOR|PASS_B)";
my $bool2 = qr"\.(?<bool2>AND|OR|XOR)";
my $func  = qr"\.(?<func>COS|SIN|EX2|LG2|RCP|RSQ|RCP64H|RSQ64H)";
my $rro   = qr"\.(?<func>SINCOS|EX2)";
my $add3  = qr"(?:\.(?<type>X|RS|LS))?";
my $lopz  = qr"(?:\.(?<z>NZ|Z) $p48,|(?<noz>))"o;
my $X     = qr"(?<X>\.X)?";
my $tld   = qr"(?<NODEP>NODEP\.)?(?:(?<reuse1>T)|(?<reuse2>P))";
my $chnls = qr"(?<chnls>R|RGBA)";
my $sr    = qr"SR_(?<sr>\S+)";
my $shf   = qr"(?<W>\.W)?(?:\.(?<type>U64|S64))?(?<HI>\.HI)?";
my $xmad  = qr"(?:\.(?<type1>U16|S16))?(?:\.(?<type2>U16|S16))?(?:\.(?<mode>MRG|PSL|CHI|CLO|CSFU))?(?<CBCC>\.CBCC)?";
my $xmadc = qr"(?:\.(?<type1>U16|S16))?(?:\.(?<type2>U16|S16))?(?:\.(?<modec>MRG|PSL|CHI|CLO|CSFU))?(?<CBCC>\.CBCC)?";
my $vmad8 = qr"\.(?<sign1>[SU])(?<size1>8|16)\.(?<sign2>[SU])(?<size2>8|16)(?<PO>\.PO)?(?<SHR_7>\.SHR_7)?(?<SHR_15>\.SHR_15)?(?<SAT>\.SAT)?";
my $vmad16= qr"\.(?<sign1>[SU])(?<size1>16)\.(?<sign2>[SU])(?<size2>16)";
my $hilo  = qr"(?:\.(?<mode>XHI|XLO))?";
my $vaddType = qr"(?:\.(?<UD>UD))?(?:\.(?<SD>SD))?(?:\.(?<sign1>[SU])(?<size1>8|16|32))?(?:\.(?<sign2>[SU])(?<size2>8|16|32))?";
my $vaddMode = qr"(?:\.(?<mode>MRG_16[HL]|MRG_8B[0-3]|ACC|MIN|MAX))?";
my $vmnmx = qr"(?:\.(?<MX>MX))?";
my $x2x   = qr"\.(?<destSign>F|U|S)(?<destWidth>8|16|32|64)\.(?<srcSign>F|U|S)(?<srcWidth>8|16|32|64)";
my $prmt  = qr"(?:\.(?<mode>F4E|B4E|RC8|ECL|ECR|RC16))?";
my $shfl  = qr"\.(?<mode>IDX|UP|DOWN|BFLY)";
my $bar   = qr"\.(?<mode>SYNC|ARV|RED)(?:\.(?<red>POPC|AND|OR))? (?:$i8w4|$r8)(?:, (?:$i20w12|$r20))?(?(<r20>)|(?<nor20>))(?(<red>), $p39|(?<nop39>))"o;
my $b2r   = qr"\.RESULT $r0(?:, $p45|(?<nop45>))"o;
my $dbar  = qr"(?<SB>SB0|SB1|SB2|SB3|SB4|SB5)";
my $dbar2 = qr" {(?<db5>5)?,?(?<db4>4)?,?(?<db3>3)?,?(?<db2>2)?,?(?<db1>1)?,?(?<db0>0)?}";
my $mbar  = qr"\.(?<mode>CTA|GL|SYS)";
my $addr  = qr"\[(?:(?<r8>$reg)|(?<nor8>))(?:\s*\+?\s*$i20w24)?\]"o;
my $addr2 = qr"\[(?:(?<r8>$reg)|(?<nor8>))(?:\s*\+?\s*$i28w20)?\]"o;
my $ldc   = qr"c\[(?<c36>$hex)\]\s*$addr"o;
my $atom  = qr"(?<E>\.E)?(?:\.(?<mode>ADD|MIN|MAX|INC|DEC|AND|OR|XOR|EXCH|CAS))(?<type>|\.S32|\.U64|\.F(?:16x2|32)\.FTZ\.RN|\.S64|\.64)";
my $vote  = qr"\.(?<mode>ALL|ANY|EQ)"o;
my $memType  = qr"(?<type>\.U8|\.S8|\.U16|\.S16||\.32|\.64|\.128)";
my $memCache = qr"(?<E>\.E)?(?<U>\.U)?(?:\.(?<cache>CG|CI|CS|CV|IL|WT))?";



# class: hardware resource that shares characteristics with types
# lat  : pipeline depth where relevent, placeholder for memory ops
# blat : barrier latency, typical fetch time for memory operations. Highly variable.
# rlat : operand read latency for memory ops
# rhold: clock cycles that a memory op typically holds onto a register before it's free to be written by another op.
# tput : throughput, clock cycles an op takes when two ops of the same class are issued in succession.
# dual : whether this instruction type can be dual issued
# reuse: whether this instruction type accepts register reuse flags.

# Some of these values are guesses and need to be updated from micro benchmarks.
# We may need to split these classes up further.
my $s2rT  = {class => 's2r',   lat => 2,   blat => 25,  rlat => 0, rhold => 0,  tput => 1,   dual => 0, reuse => 0};
my $smemT = {class => 'mem',   lat => 2,   blat => 30,  rlat => 2, rhold => 20, tput => 1,   dual => 1, reuse => 0};
my $gmemT = {class => 'mem',   lat => 2,   blat => 200, rlat => 4, rhold => 20, tput => 1,   dual => 1, reuse => 0};
my $x32T  = {class => 'x32',   lat => 6,   blat => 0,   rlat => 0, rhold => 0,  tput => 1,   dual => 0, reuse => 1};
my $x64T  = {class => 'x64',   lat => 2,   blat => 128, rlat => 0, rhold => 0,  tput => 128, dual => 0, reuse => 1};
my $shftT = {class => 'shift', lat => 6,   blat => 0,   rlat => 0, rhold => 0,  tput => 2,   dual => 0, reuse => 1};
my $cmpT  = {class => 'cmp',   lat => 13,  blat => 0,   rlat => 0, rhold => 0,  tput => 2,   dual => 0, reuse => 1};
my $qtrT  = {class => 'qtr',   lat => 8,   blat => 0,   rlat => 4, rhold => 0,  tput => 1,   dual => 1, reuse => 0};
my $rroT  = {class => 'rro',   lat => 2,   blat => 0,   rlat => 0, rhold => 0,  tput => 1,   dual => 0, reuse => 0};
my $voteT = {class => 'vote',  lat => 2,   blat => 0,   rlat => 0, rhold => 0,  tput => 1,   dual => 0, reuse => 0};


# Create map of op names to rules
our %grammar =
(
    #Floating Point Instructions
    FADD     => [ { type => $x32T,  code => 0x5c58000000000000, rule => qr"^$pred?FADD$ftz$rnd$sat $r0, $r8, $fcr20;"o,               } ],
    FADD32I  => [ { type => $x32T,  code => 0x0800000000000000, rule => qr"^$pred?FADD32I$ftz $r0, $r8, $f20w32;"o,                   } ],
    FCHK     => [ { type => $x32T,  code => 0x5c88000000000000, rule => qr"^$pred?FCHK\.DIVIDE $p0, $r8, $r20;"o,                     } ], #Partial?
    FCMP     => [ { type => $cmpT,  code => 0x5ba0000000000000, rule => qr"^$pred?FCMP$fcmp$ftz $r0, $r8, $fcr20, $r39;"o,            } ],
    FFMA     => [
                  { type => $x32T,  code => 0x5980000000000000, rule => qr"^$pred?FFMA$ftz$rnd$sat $r0, $r8, $fcr20, $r39;"o,         },
                  { type => $x32T,  code => 0x5980000000000000, rule => qr"^$pred?FFMA$ftz$rnd$sat $r0, $r8, $r39s20, $c20s39;"o,     },
                ],
    FMNMX    => [ { type => $shftT, code => 0x5c60000000000000, rule => qr"^$pred?FMNMX$ftz $r0, $r8, $fcr20, $p39;"o,                } ],
    FMUL     => [ { type => $x32T,  code => 0x5c68000000000000, rule => qr"^$pred?FMUL$ftz$rnd$sat $r0, $r8, $fcr20;"o,               } ],
    FMUL32I  => [ { type => $x32T,  code => 0x1e00000000000000, rule => qr"^$pred?FMUL32I$ftz $r0, $r8, $f20w32;"o,                   } ],
    FSET     => [ { type => $shftT, code => 0x5800000000000000, rule => qr"^$pred?FSET$fcmp$ftz$bool $r0, $r8, $fcr20, $p39;"o,       } ],
    FSETP    => [ { type => $cmpT,  code => 0x5bb0000000000000, rule => qr"^$pred?FSETP$fcmp$ftz$bool $p3, $p0, $r8, $fcr20, $p39;"o, } ],
    MUFU     => [ { type => $qtrT,  code => 0x5080000000000000, rule => qr"^$pred?MUFU$func $r0, $r8;"o,                              } ],
    RRO      => [ { type => $rroT,  code => 0x5c90000000000000, rule => qr"^$pred?RRO$rro $r0, $r20;"o,                               } ],
    DADD     => [ { type => $x64T,  code => 0x5c70000000000000, rule => qr"^$pred?DADD$rnd $r0, $r8, $dr20;"o,                        } ],
    DFMA     => [ { type => $x64T,  code => 0x5b70000000000000, rule => qr"^$pred?DFMA$rnd $r0, $r8, $dr20, $r39;"o,                  } ],
    DMNMX    => [ { type => $cmpT,  code => 0x5c50000000000000, rule => qr"^$pred?DMNMX $r0, $r8, $dr20, $p39;"o,                     } ],
    DMUL     => [ { type => $x64T,  code => 0x5c80000000000000, rule => qr"^$pred?DMUL$rnd $r0, $r8, $dr20;"o,                        } ],
    DSET     => [ { type => $cmpT,  code => 0x5900000000000000, rule => qr"^$pred?DSET$fcmp$bool $r0, $r8, $dr20, $p39;"o,            } ],
    DSETP    => [ { type => $cmpT,  code => 0x5b80000000000000, rule => qr"^$pred?DSETP$fcmp$bool $p3, $p0, $r8, $dr20, $p39;"o,      } ],
    FSWZADD  => [ { type => $x32T,  code => 0x0000000000000000, rule => qr"^$pred?FSWZADD[^;]*;"o,                                    } ], #TODO

    HADD2     => [ { type => $x32T,  code => 0x5d10000000000000, rule => qr"^$pred?HADD2$ftz $r0, $r8, $r20;"o,               } ],
    HMUL2     => [ { type => $x32T,  code => 0x5d08000000000000, rule => qr"^$pred?HMUL2$ftz $r0, $r8, $r20;"o,               } ],
    HFMA2     => [ { type => $x32T,  code => 0x5d00000000000000, rule => qr"^$pred?HFMA2$ftz $r0, $r8, $r20, $r39;"o,         } ],
    HSETP2    => [ { type => $cmpT,  code => 0x5d20000000000000, rule => qr"^$pred?HSETP2$fcmp$bool $p3, $p0, $r8, $fcr20, $p39;"o, } ], #Partial

    #Integer Instructions
    BFE       => [ { type => $shftT,  code => 0x5c01000000000000, rule => qr"^$pred?BFE$u32 $r0, $r8, $icr20;"o,                          } ],
    BFI       => [ { type => $shftT,  code => 0x5bf0000000000000, rule => qr"^$pred?BFI $r0, $r8, $ir20, $cr39;"o,                        } ],
    FLO       => [ { type => $s2rT,   code => 0x5c30000000000000, rule => qr"^$pred?FLO\.U32 $r0, $icr20;"o,                              } ],
    IADD      => [ { type => $x32T,   code => 0x5c10000000000000, rule => qr"^$pred?IADD$sat$X $r0cc, $r8, $icr20;"o,                         } ],
    IADD32I   => [ { type => $x32T,   code => 0x1c00000000000000, rule => qr"^$pred?IADD32I$X $r0cc, $r8, $i20w32;"o,                         } ],
    IADD3     => [ { type => $x32T,   code => 0x5cc0000000000000, rule => qr"^$pred?IADD3$add3 $r0cc, $r8, $icr20, $r39;"o,                 } ],
    ICMP      => [ { type => $cmpT,   code => 0x5b41000000000000, rule => qr"^$pred?ICMP$icmp$u32 $r0, $r8, $icr20, $r39;"o,              } ],
    IMNMX     => [ { type => $shftT,  code => 0x5c21000000000000, rule => qr"^$pred?IMNMX$u32$hilo $r0cc, $r8, $icr20, $p39;"o,                  } ],
    ISET      => [ { type => $shftT,  code => 0x5b51000000000000, rule => qr"^$pred?ISET$icmp$u32$X$bool $r0, $r8, $icr20, $p39;"o,       } ],
    ISETP     => [ { type => $cmpT,   code => 0x5b61000000000000, rule => qr"^$pred?ISETP$icmp$u32$X$bool $p3, $p0, $r8, $icr20, $p39;"o, } ],
    ISCADD    => [ { type => $shftT,  code => 0x5c18000000000000, rule => qr"^$pred?ISCADD $r0, $r8, $icr20, $i39w8;"o,                   } ],
    ISCADD32I => [ { type => $shftT,  code => 0x1400000000000000, rule => qr"^$pred?ISCADD32I $r0, $r8, $i20w32, $i53w5;"o,               } ],
    LEA       => [
                   { type => $cmpT,   code => 0x5bd0000000000000, rule => qr"^$pred?LEA $p48, $r0cc, $r8, $icr20;"o,                      },
                   { type => $shftT,  code => 0x5bd7000000000000, rule => qr"^$pred?LEA $r0cc, $r8, $icr20, $i39w8;"o,                    },
                   { type => $shftT,  code => 0x5bdf004000000000, rule => qr"^$pred?LEA\.HI$X $r0cc, $r8, $r20, $r39, $i28w8;"o,          },
                   { type => $shftT,  code => 0x0a07000000000000, rule => qr"^$pred?LEA\.HI$X $r0cc, $r8, $c20, $r39, $i51w5;"o,          },
                 ],
    LOP       => [ { type => $x32T,   code => 0x5c40000000000000, rule => qr"^$pred?LOP$bool$lopz $r0, $r8, (?<INV>~)?$icr20(?<INV>\.INV)?;"o, } ],
    LOP32I    => [ { type => $x32T,   code => 0x0400000000000000, rule => qr"^$pred?LOP32I$bool $r0, $r8, $i20w32;"o,                     } ],
    LOP3      => [
                   { type => $x32T,   code => 0x5be7000000000000, rule => qr"^$pred?LOP3\.LUT $r0, $r8, $r20, $r39, $i28w8;"o,            },
                   { type => $x32T,   code => 0x3c00000000000000, rule => qr"^$pred?LOP3\.LUT $r0, $r8, $i20, $r39, $i48w8;"o,            },
                 ],
    POPC      => [ { type => $s2rT,   code => 0x5c08000000000000, rule => qr"^$pred?POPC $r0, $r20;"o,                                    } ],
    SHF       => [
                   { type => $shftT,  code => 0x5bf8000000000000, rule => qr"^$pred?SHF\.L$shf $r0, $r8, $ir20, $r39;"o,                  },
                   { type => $shftT,  code => 0x5cf8000000000000, rule => qr"^$pred?SHF\.R$shf $r0, $r8, $ir20, $r39;"o,                  },
                 ],
    SHL       => [ { type => $shftT,  code => 0x5c48000000000000, rule => qr"^$pred?SHL(?<W>\.W)? $r0, $r8, $icr20;"o,                    } ],
    SHR       => [ { type => $shftT,  code => 0x5c29000000000000, rule => qr"^$pred?SHR$u32 $r0, $r8, $icr20;"o,                          } ],
    XMAD      => [
                   { type => $x32T,   code => 0x5b00000000000000, rule => qr"^$pred?XMAD$xmad $r0cc, $r8, $ir20, $r39;"o,                 },
                   { type => $x32T,   code => 0x5900000000000000, rule => qr"^$pred?XMAD$xmad $r0cc, $r8, $r39s20, $c20s39;"o,            },
                   { type => $x32T,   code => 0x5e00000000000000, rule => qr"^$pred?XMAD$xmadc $r0cc, $r8, $c20x, $r39;"o,                  },
                 ],
    # XMAD replaces these
    IMAD      => [ { type => $x32T,   code => 0x0000000000000000, rule => qr"^$pred?IMAD[^;]*;"o,   } ], #TODO
    IMADSP    => [ { type => $x32T,   code => 0x0000000000000000, rule => qr"^$pred?IMADSP[^;]*;"o, } ], #TODO
    IMUL      => [ { type => $x32T,   code => 0x0000000000000000, rule => qr"^$pred?IMUL[^;]*;"o,   } ], #TODO

    #Conversion Instructions
    F2F => [ { type => $qtrT,  code => 0x5ca8000000000000, rule => qr"^$pred?F2F$ftz$x2x$rnd$round$sat $r0, $cr20;"o, } ],
    F2I => [ { type => $qtrT,  code => 0x5cb0000000000000, rule => qr"^$pred?F2I$ftz$x2x$round $r0, $cr20;"o,         } ],
    I2F => [ { type => $qtrT,  code => 0x5cb8000000000000, rule => qr"^$pred?I2F$x2x$rnd $r0, $cr20;"o,               } ],
    I2I => [ { type => $qtrT,  code => 0x5ce0000000000000, rule => qr"^$pred?I2I$x2x$sat $r0, $cr20;"o,               } ],

    #Movement Instructions
    MOV    => [ { type => $x32T,  code => 0x5c98078000000000, rule => qr"^$pred?MOV $r0, $icr20;"o,                   } ],
    MOV32I => [ { type => $x32T,  code => 0x010000000000f000, rule => qr"^$pred?MOV32I $r0, (?:$i20w32|$f20w32);"o,   } ],
    PRMT   => [ { type => $x32T,  code => 0x5bc0000000000000, rule => qr"^$pred?PRMT$prmt $r0, $r8, $icr20, $cr39;"o, } ],
    SEL    => [ { type => $x32T,  code => 0x5ca0000000000000, rule => qr"^$pred?SEL $r0, $r8, $icr20, $p39;"o,        } ],
    SHFL   => [ { type => $smemT, code => 0xef10000000000000, rule => qr"^$pred?SHFL$shfl $p48, $r0, $r8, (?:$i20w8|$r20), (?:$i34w13|$r39);"o, } ],

    #Predicate/CC Instructions
    PSET   => [ { type => $cmpT,  code => 0x5088000000000000, rule => qr"^$pred?PSET$bool2$bool $r0, $p12, $p29, $p39;"o,       } ],
    PSETP  => [ { type => $cmpT,  code => 0x5090000000000000, rule => qr"^$pred?PSETP$bool2$bool $p3, $p0, $p12, $p29, $p39;"o, } ],
    CSET   => [ { type => $x32T,  code => 0x0000000000000000, rule => qr"^$pred?CSET[^;]*;"o,  } ], #TODO
    CSETP  => [ { type => $x32T,  code => 0x0000000000000000, rule => qr"^$pred?CSETP[^;]*;"o, } ], #TODO
    P2R    => [ { type => $x32T,  code => 0x38e8000000000000, rule => qr"^$pred?P2R $r0, PR, $r8, $i20w7;"o,   } ],
    R2P    => [ { type => $shftT, code => 0x38f0000000000000, rule => qr"^$pred?R2P PR, $r8, $i20w7;"o,   } ],

    #Texture Instructions
    # Handle the commonly used 1D texture functions.. but save the others for later
    TLD    => [ { type => $gmemT, code => 0xdd38000000000000, rule => qr"^$pred?TLD\.B\.LZ\.$tld $r0, $r8, $r20, $hex, \dD, $i31w4;"o, } ], #Partial
    TLDS   => [ { type => $gmemT, code => 0xda0000000ff00000, rule => qr"^$pred?TLDS\.LZ\.$tld $r28, $r0, $r8, $i36w20, \dD, $chnls;"o,} ], #Partial
    TEX    => [ { type => $gmemT, code => 0x0000000000000000, rule => qr"^$pred?TEX[^;]*;"o,   } ], #TODO
    TLD4   => [ { type => $gmemT, code => 0x0000000000000000, rule => qr"^$pred?TLD4[^;]*;"o,  } ], #TODO
    TXQ    => [ { type => $gmemT, code => 0x0000000000000000, rule => qr"^$pred?TXQ[^;]*;"o,   } ], #TODO
    TEXS   => [ { type => $gmemT, code => 0x0000000000000000, rule => qr"^$pred?TEXS[^;]*;"o,  } ], #TODO
    TLD4S  => [ { type => $gmemT, code => 0x0000000000000000, rule => qr"^$pred?TLD4S[^;]*;"o, } ], #TODO

    #Compute Load/Store Instructions
    LD     => [ { type => $gmemT, code => 0x8000000000000000, rule => qr"^$pred?LD$memCache$memType $r0, $addr, $p58;"o,      } ],
    ST     => [ { type => $gmemT, code => 0xa000000000000000, rule => qr"^$pred?ST$memCache$memType $addr, $r0, $p58;"o,      } ],
    LDG    => [ { type => $gmemT, code => 0xeed0000000000000, rule => qr"^$pred?LDG$memCache$memType $r0, $addr;"o,           } ],
    STG    => [ { type => $gmemT, code => 0xeed8000000000000, rule => qr"^$pred?STG$memCache$memType $addr, $r0;"o,           } ],
    LDS    => [ { type => $smemT, code => 0xef48000000000000, rule => qr"^$pred?LDS$memCache$memType $r0, $addr;"o,           } ],
    STS    => [ { type => $smemT, code => 0xef58000000000000, rule => qr"^$pred?STS$memCache$memType $addr, $r0;"o,           } ],
    LDL    => [ { type => $gmemT, code => 0xef40000000000000, rule => qr"^$pred?LDL$memCache$memType $r0, $addr;"o,           } ],
    STL    => [ { type => $gmemT, code => 0xef50000000000000, rule => qr"^$pred?STL$memCache$memType $addr, $r0;"o,           } ],
    LDC    => [ { type => $gmemT, code => 0xef90000000000000, rule => qr"^$pred?LDC$memCache$memType $r0, $ldc;"o,            } ],
    # Note for ATOM(S).CAS operations the last register needs to be in sequence with the second to last (as it's not encoded).
    ATOM   => [ { type => $gmemT, code => 0xed00000000000000, rule => qr"^$pred?ATOM$atom $r0, $addr2, $r20(?:, $r39a)?;"o,   } ],
    ATOMS  => [ { type => $smemT, code => 0xec00000000000000, rule => qr"^$pred?ATOMS$atom $r0, $addr2, $r20(?:, $r39a)?;"o,  } ],
    RED    => [ { type => $gmemT, code => 0xebf8000000000000, rule => qr"^$pred?RED$atom $addr2, $r0;"o,                      } ],
    CCTL   => [ { type => $x32T,  code => 0x5c88000000000000, rule => qr"^$pred?CCTL[^;]*;"o,  } ], #TODO
    CCTLL  => [ { type => $x32T,  code => 0x5c88000000000000, rule => qr"^$pred?CCTLL[^;]*;"o, } ], #TODO
    CCTLT  => [ { type => $x32T,  code => 0x5c88000000000000, rule => qr"^$pred?CCTLT[^;]*;"o, } ], #TODO

    #Surface Memory Instructions (haven't gotten to these yet..)
    SUATOM => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?SUATOM[^;]*;"o, } ], #TODO
    SULD   => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?SULD[^;]*;"o,   } ], #TODO
    SURED  => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?SURED[^;]*;"o,  } ], #TODO
    SUST   => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?SUST[^;]*;"o,   } ], #TODO

    #Control Instructions
    BRA    => [
                { type => $x32T, code => 0xe24000000000000f, rule => qr"^$pred?BRA(?<U>\.U)? $i20w24;"o,         },
                { type => $x32T, code => 0xe240000000000002, rule => qr"^$pred?BRA(?<U>\.U)? CC\.EQ, $i20w24;"o, },
              ],
    BRX    => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?BRX[^;]*;"o,                      } ], #TODO
    JMP    => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?JMP[^;]*;"o,                      } ], #TODO
    JMX    => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?JMX[^;]*;"o,                      } ], #TODO
    SSY    => [ { type => $x32T, code => 0xe290000000000000, rule => qr"^$noPred?SSY $i20w24;"o,                 } ],
    SYNC   => [ { type => $x32T, code => 0xf0f800000000000f, rule => qr"^$pred?SYNC;"o,                          } ],
    CAL    => [ { type => $x32T, code => 0xe260000000000040, rule => qr"^$noPred?CAL $i20w24;"o,                 } ],
    JCAL   => [ { type => $x32T, code => 0xe220000000000040, rule => qr"^$noPred?JCAL $i20w24;"o,                } ],
    PRET   => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?PRET[^;]*;"o,                     } ], #TODO
    RET    => [ { type => $x32T, code => 0xe32000000000000f, rule => qr"^$pred?RET;"o,                           } ],
    BRK    => [ { type => $x32T, code => 0xe34000000000000f, rule => qr"^$pred?BRK;"o,                           } ],
    PBK    => [ { type => $x32T, code => 0xe2a0000000000000, rule => qr"^$noPred?PBK $i20w24;"o,                 } ],
    CONT   => [ { type => $x32T, code => 0xe35000000000000f, rule => qr"^$pred?CONT;"o,                          } ],
    PCNT   => [ { type => $x32T, code => 0xe2b0000000000000, rule => qr"^$noPred?PCNT $i20w24;"o,                } ],
    EXIT   => [ { type => $x32T, code => 0xe30000000000000f, rule => qr"^$pred?EXIT;"o,                          } ],
    PEXIT  => [ { type => $x32T, code => 0x0000000000000000, rule => qr"^$pred?PEXIT[^;]*;"o,                    } ], #TODO
    BPT    => [ { type => $x32T, code => 0xe3a00000000000c0, rule => qr"^$noPred?BPT\.TRAP $i20w24;"o,           } ],

    #Miscellaneous Instructions
    NOP    => [ { type => $x32T,  code => 0x50b0000000000f00, rule => qr"^$pred?NOP;"o,                                     } ],
    CS2R   => [ { type => $x32T,  code => 0x50c8000000000000, rule => qr"^$pred?CS2R $r0, $sr;"o,                           } ],
    S2R    => [ { type => $s2rT,  code => 0xf0c8000000000000, rule => qr"^$pred?S2R $r0, $sr;"o,                            } ],
    B2R    => [ { type => $x32T,  code => 0xf0b800010000ff00, rule => qr"^$pred?B2R$b2r;"o,                                 } ],
    BAR    => [ { type => $gmemT, code => 0xf0a8000000000000, rule => qr"^$pred?BAR$bar;"o,                                 } ],
    DEPBAR => [
                { type => $gmemT, code => 0xf0f0000000000000, rule => qr"^$pred?DEPBAR$icmp $dbar, $i20w6;"o, },
                { type => $gmemT, code => 0xf0f0000000000000, rule => qr"^$pred?DEPBAR$dbar2;"o,              },
              ],
    MEMBAR => [ { type => $x32T,  code => 0xef98000000000000, rule => qr"^$pred?MEMBAR$mbar;"o,                             } ],
    VOTE   => [ { type => $voteT, code => 0x50d8000000000000, rule => qr"^$pred?VOTE$vote (?:$r0, |(?<nor0>))$p45, $p39;"o, } ],
    R2B    => [ { type => $x32T,  code => 0x0000000000000000, rule => qr"^$pred?R2B[^;]*;"o,                                } ], #TODO

    #Video Instructions... Need to finish
    VADD   => [   { type => $shftT, code => 0x2044000000000000, rule => qr"^$pred?VADD$vaddType$sat$vaddMode $r0, $r8, $r20, $r39;"o, } ], #Partial 0x2044000000000000
    VMAD   => [
                  { type => $x32T,  code => 0x5f04000000000000, rule => qr"^$pred?VMAD$vmad16 $r0, $r8, $r20, $r39;"o, },
                  { type => $shftT, code => 0x5f04000000000000, rule => qr"^$pred?VMAD$vmad8 $r0, $r8, $r20, $r39;"o, },
              ],
    VABSDIFF => [ { type => $shftT, code => 0x5427000000000000, rule => qr"^$pred?VABSDIFF$vaddType$sat$vaddMode $r0, $r8, $r20, $r39;"o, } ], #Partial 0x2044000000000000
    VMNMX    => [ { type => $shftT, code => 0x3a44000000000000, rule => qr"^$pred?VMNMX$vaddType$vmnmx$sat$vaddMode $r0, $r8, $r20, $r39;"o, } ], #Partial 0x2044000000000000

    VSET => [ { type => $shftT, code => 0x4004000000000000, rule => qr"^$pred?VSET$icmp$vaddType$vaddMode $r0, $r8, $r20, $r39;"o, } ], #Partial 0x2044000000000000
);

# Create map of capture groups to op code flags that need to be added (or removed)
my @flags = grep /\S/, split "\n", q{;

BFE, BFI, FLO, IADD, IADD3, ICMP, IMNMX, ISCADD, ISET, ISETP, LEA, LOP, LOP3, MOV, PRMT, SEL, SHF, SHL, SHR, XMAD
0x0100000000000000 neg

FADD, FCMP, FFMA, FMNMX, FMUL, FSET, FSETP, DADD, DFMA, DMNMX, DMUL, DSET, DSETP
0x0100000000000000 neg

PSET, PSETP
0x0000000000008000 p12not
0x0000000100000000 p29not

FMNMX, FSET, FSETP, DMNMX, DSET, DSETP, IMNMX, ISET, ISETP, SEL, PSET, PSETP, BAR, VOTE
0x0000040000000000 p39not

IADD, IADD3, XMAD, LEA, IMNMX
0x0000800000000000 CC

IADD32I
0x0010000000000000 CC

LEA
0x0000000000000000 X

SHF
0x0004000000000000 W
0x0001000000000000 HI

SHF: type
0x0000004000000000 U64
0x0000006000000000 S64

SHR, IMNMX, ISETP, ISET, ICMP, BFE
0x0001000000000000 U32

SHL
0x0000008000000000 W

SHFL
0x0000000010000000 i20w8
0x0000000020000000 i34w13

SHFL: mode
0x0000000000000000 IDX
0x0000000040000000 UP
0x0000000080000000 DOWN
0x00000000c0000000 BFLY

IMNMX: mode
0x0000080000000000 XLO
0x0000180000000000 XHI

ISETP, ISET, ICMP: cmp
0x0002000000000000 LT
0x0004000000000000 EQ
0x0006000000000000 LE
0x0008000000000000 GT
0x000a000000000000 NE
0x000c000000000000 GE

ISETP, ISET, PSETP, PSET: bool
0x0000000000000000 AND
0x0000200000000000 OR
0x0000400000000000 XOR

PSETP, PSET: bool2
0x0000000000000000 AND
0x0000000001000000 OR
0x0000000002000000 XOR

ISETP, ISET
0x0000080000000000 X

LOP: bool
0x0000000000000000 AND
0x0000020000000000 OR
0x0000040000000000 XOR
0x0000060000000000 PASS_B

LOP:
0x0000010000000000 INV

LOP: z
0x0000200000000000 Z
0x0000300000000000 NZ

LOP
0x0007000000000000 noz

LOP32I: bool
0x0000000000000000 AND
0x0020000000000000 OR
0x0040000000000000 XOR

PRMT: mode
0x0001000000000000 F4E
0x0002000000000000 B4E
0x0003000000000000 RC8
0x0004000000000000 ECL
0x0005000000000000 ECR
0x0006000000000000 RC16

XMAD: type1
0x0000000000000000 U16
0x0001000000000000 S16

XMAD: type2
0x0000000000000000 U16
0x0002000000000000 S16

XMAD: mode
0x0000002000000000 MRG
0x0000001000000000 PSL
0x0008000000000000 CHI
0x0004000000000000 CLO
0x000c000000000000 CSFU

XMAD: modec
0x0004000000000000 CLO
0x0008000000000000 CHI
0x000c000000000000 CSFU
0x0040000000000000 X
0x0080000000000000 PSL
0x0100000000000000 MRG

XMAD
0x0010000000000000 CBCC

XMAD: r8part
0x0000000000000000 H0
0x0020000000000000 H1

XMAD: r20part
0x0000000000000000 H0
0x0000000800000000 H1

XMAD: r20partx
0x0000000000000000 H0
0x0010000000000000 H1

XMAD: r39part
0x0000000000000000 H0
0x0010000000000000 H1

VMAD, VADD, VABSDIFF, VMNMX, VSET: r8part
0x0000000000000000 B0
0x0000001000000000 B1
0x0000002000000000 B2
0x0000003000000000 B3
0x0000001000000000 H1
0x0000000000000000 H0

VMAD, VADD, VABSDIFF, VMNMX, VSET: r20part
0x0000000000000000 B0
0x0000000010000000 B1
0x0000000020000000 B2
0x0000000030000000 B3
0x0000000010000000 H1
0x0000000000000000 H0

VMAD
0x0040000000000000 r8neg
0x0020000000000000 r39neg
0x0008000000000000 SHR_7
0x0010000000000000 SHR_15
0x0060000000000000 PO
0x0080000000000000 SAT

VMNMX
0x0100000000000000 MX

VADD, VABSDIFF, VMNMX
0x0080000000000000 SAT
0x0040000000000000 UD
0x0040000000000000 SD

VSET: cmp
0x0040000000000000 LT
0x0080000000000000 EQ
0x00c0000000000000 LE
0x0100000000000000 GT
0x0140000000000000 NE
0x0180000000000000 GE

VADD, VSET: mode
0x0020000000000000 ACC
0x0028000000000000 MIN
0x0030000000000000 MAX
0x0000000000000000 MRG_16H
0x0008000000000000 MRG_16L
0x0010000000000000 MRG_8B0
0x0000000000000000 MRG_8B1
0x0018000000000000 MRG_8B2
0x0000000000000000 MRG_8B3

VABSDIFF: mode
0x0003000000000000 ACC
0x000b000000000000 MIN
0x0013000000000000 MAX
0x0023000000000000 MRG_16H
0x002b000000000000 MRG_16L
0x0033000000000000 MRG_8B0
0x0000000000000000 MRG_8B1
0x003b000000000000 MRG_8B2
0x0000000000000000 MRG_8B3

VMNMX: mode
0x0020000000000000 ACC
0x0028000000000000 MIN
0x0030000000000000 MAX
0x0000000000000000 MRG_16H
0x0008000000000000 MRG_16L
0x0010000000000000 MRG_8B0
0x0000000000000000 MRG_8B1
0x0018000000000000 MRG_8B2
0x0000000000000000 MRG_8B3

VMAD, VADD, VABSDIFF, VMNMX, VSET: sign1
0x0000000000000000 U
0x0001000000000000 S

VMAD, VADD, VABSDIFF, VMNMX, VSET: sign2
0x0000000000000000 U
0x0002000000000000 S

VMAD, VADD, VABSDIFF, VMNMX, VSET: size1
0x0000000000000000 8
0x0000004000000000 16
0x0000006000000000 32

VMAD, VADD, VABSDIFF, VMNMX, VSET: size2
0x0000000000000000 8
0x0000000040000000 16
0x0000000060000000 32

IADD3: type
0x0001000000000000 X
0x0000002000000000 RS
0x0000004000000000 LS

IADD3: r8part
0x0000000000000000 H0
0x0000001000000000 H1

IADD3: r20part
0x0000000080000000 H0

IADD3: r39part
0x0000000200000000 H0

IADD3
0x0008000000000000 r8neg
0x0004000000000000 r20neg
0x0002000000000000 r39neg

IADD
0x0000080000000000 X
0x0004000000000000 SAT

IADD, ISCADD
0x0002000000000000 r8neg
0x0001000000000000 r20neg

IADD32I
0x0100000000000000 r8neg
0x0020000000000000 X

DEPBAR: SB
0x0000000000000000 SB0
0x0000000004000000 SB1
0x0000000008000000 SB2
0x000000000c000000 SB3
0x0000000010000000 SB4
0x0000000014000000 SB5

DEPBAR: cmp
0x0000000020000000 LE

DEPBAR
0x0000000000000001 db0
0x0000000000000002 db1
0x0000000000000004 db2
0x0000000000000008 db3
0x0000000000000010 db4
0x0000000000000020 db5

F2F, F2I, I2F, I2I: destWidth
0x0000000000000000 8
0x0000000000000100 16
0x0000000000000200 32
0x0000000000000300 64

F2F, F2I, I2F, I2I: srcWidth
0x0000000000000000 8
0x0000000000000400 16
0x0000000000000800 32
0x0000000000000c00 64

F2F, F2I, I2F, I2I: destSign
0x0000000000000000 F
0x0000000000000000 U
0x0000000000001000 S

F2F, F2I, I2F, I2I: srcSign
0x0000000000000000 F
0x0000000000000000 U
0x0000000000002000 S

F2I, I2F, I2I: r20part
0x0000000000000000 H0
0x0000040000000000 H1
0x0000000000000000 B0
0x0000020000000000 B1
0x0000040000000000 B2
0x0000060000000000 B3

F2F: r20part
0x0000000000000000 H0
0x0000020000000000 H1

F2F: round
0x0000040000000000 ROUND
0x0000048000000000 FLOOR
0x0000050000000000 CEIL
0x0000058000000000 TRUNC

F2I: round
0x0000000000000000 ROUND
0x0000008000000000 FLOOR
0x0000010000000000 CEIL
0x0000018000000000 TRUNC

HADD2, HMUL2: r8part
0x0001000000000000 H0_H0
0x0000000000000000 H1_H1

HFMA2: r20part
0x0000000020000000 H0_H0
0x0000000030000000 H1_H1

FADD, DADD, FMUL, DMUL, F2F, I2F: rnd
0x0000000000000000 RN
0x0000008000000000 RM
0x0000010000000000 RP
0x0000018000000000 RZ

DFMA: rnd
0x0000000000000000 RN
0x0004000000000000 RM
0x0008000000000000 RP
0x000c000000000000 RZ

FFMA: rnd
0x0000000000000000 RN
0x0008000000000000 RM
0x0010000000000000 RP
0x0018000000000000 RZ

FFMA
0x0020000000000000 FTZ

F2F, F2I, FADD, FMUL, FMNMX
0x0000100000000000 FTZ

FADD32I
0x0080000000000000 FTZ

FMUL32I
0x0020000000000000 FTZ

FSET
0x0080000000000000 FTZ

FSETP, FCMP
0x0000800000000000 FTZ

HADD2, HMUL2
0x0000008000000000 FTZ

HFMA2
0x0000002000000000 FTZ

FADD, FFMA, FMUL, F2F, I2I
0x0004000000000000 SAT

FADD, DADD, FMNMX, DMNMX, MUFU
0x0001000000000000 r8neg

FADD, DADD, FMNMX, DMNMX, RRO, F2F, F2I, I2F, I2I
0x0000200000000000 r20neg

FMUL, DMUL, FFMA, DFMA
0x0001000000000000 r20neg

FFMA, DFMA
0x0002000000000000 r39neg

FADD, DADD, FMNMX, DMNMX
0x0000400000000000 r8abs

FADD, DADD, FMNMX, DMNMX, F2F, F2I, I2F, I2I
0x0002000000000000 r20abs

FSETP, DSETP, FSET, DSET
0x0000080000000000 r8neg
0x0000000000000040 r20neg
0x0000000000000080 r8abs
0x0000100000000000 r20abs

RRO: func
0x0000000000000000 SINCOS
0x0000008000000000 EX2

MUFU: func
0x0000000000000000 COS
0x0000000000100000 SIN
0x0000000000200000 EX2
0x0000000000300000 LG2
0x0000000000400000 RCP
0x0000000000500000 RSQ
0x0000000000600000 RCP64H
0x0000000000700000 RSQ64H

FSETP, DSETP, FSET, DSET, FCMP: cmp
0x0001000000000000 .LT
0x0002000000000000 .EQ
0x0003000000000000 .LE
0x0004000000000000 .GT
0x0004000000000000
0x0005000000000000 .NE
0x0006000000000000 .GE
0x0007000000000000 .NUM
0x0008000000000000 .NAN
0x0009000000000000 .LTU
0x000a000000000000 .EQU
0x000b000000000000 .LEU
0x000c000000000000 .GTU
0x000d000000000000 .NEU
0x000e000000000000 .GEU

FSETP, DSETP, FSET, DSET: bool
0x0000000000000000 AND
0x0000200000000000 OR
0x0000400000000000 XOR

HSETP2: cmp
0x0000002800000000 .NE

HSETP2: bool
0x0000000000000000 AND

S2R: sr
0x0000000000000000 LANEID
0x0000000000200000 VIRTCFG
0x0000000000300000 VIRTID
0x0000000002100000 TID.X
0x0000000002200000 TID.Y
0x0000000002300000 TID.Z
0x0000000002500000 CTAID.X
0x0000000002600000 CTAID.Y
0x0000000002700000 CTAID.Z
0x0000000003800000 EQMASK
0x0000000003900000 LTMASK
0x0000000003a00000 LEMASK
0x0000000003b00000 GTMASK
0x0000000003c00000 GEMASK

CS2R: sr
0x0000000005000000 CLOCKLO
0x0000000005100000 CLOCKHI
0x0000000005200000 GLOBALTIMERLO
0x0000000005300000 GLOBALTIMERHI

B2R
0x0000e00000000000 nop45

BAR
0x0000100000000000 i8w4
0x0000080000000000 nor20
0x0000038000000000 nop39

BAR: mode
0x0000000000000000 SYNC
0x0000000100000000 ARV
0x0000000200000000 RED

BAR: red
0x0000000000000000 POPC
0x0000000800000000 AND
0x0000001000000000 OR

MEMBAR: mode
0x0000000000000000 CTA
0x0000000000000100 GL
0x0000000000000200 SYS

VOTE: mode
0x0000000000000000 ALL
0x0001000000000000 ANY
0x0002000000000000 EQ

VOTE
0x00000000000000ff nor0

BRA
0x0000000000000080 U

TLDS: chnls
0x0010000000000000 RGBA

TLDS
0x0002000000000000 NODEP

LD, ST, LDG, STG, LDS, STS, LDL, STL, LDC, RED, ATOM, ATOMS
0x000000000000ff00 nor8

LD, ST: type
0x0000000000000000 .U8
0x0020000000000000 .S8
0x0040000000000000 .U16
0x0060000000000000 .S16
0x0080000000000000
0x0080000000000000 .32
0x00a0000000000000 .64
0x00c0000000000000 .128

LD, ST: cache
0x0100000000000000 CG
0x0200000000000000 CS
0x0300000000000000 CV
0x0300000000000000 WT

LDG, STG, LDS, STS, LDL, STL, LDC: type
0x0000000000000000 .U8
0x0001000000000000 .S8
0x0002000000000000 .U16
0x0003000000000000 .S16
0x0004000000000000
0x0004000000000000 .32
0x0005000000000000 .64
0x0006000000000000 .128

LDG, STG: cache
0x0000400000000000 CG
0x0000800000000000 CI
0x0000800000000000 CS
0x0000c00000000000 CV
0x0000c00000000000 WT

LDL: cache
0x0000200000000000 CI

LDC: cache
0x0000100000000000 IL

LDG, STG, LDS, STS, LDL, STL, LDC
0x0000200000000000 E

LDS
0x0000100000000000 U

RED: type
0x0000000000000000
0x0000000000100000 .S32
0x0000000000200000 .U64
0x0000000000300000 .F32.FTZ.RN
0x0000000000400000 .F16x2.FTZ.RN
0x0000000000500000 .S64

RED: mode
0x0000000000000000 ADD
0x0000000000800000 MIN
0x0000000001000000 MAX
0x0000000001800000 INC
0x0000000002000000 DEC
0x0000000002800000 AND
0x0000000003000000 OR
0x0000000003800000 XOR

ATOM: type
0x0000000000000000
0x0002000000000000 .S32
0x0004000000000000 .U64
0x0006000000000000 .F32.FTZ.RN
0x0008000000000000 .F16x2.FTZ.RN
0x000a000000000000 .S64
0x0002000000000000 .64

ATOM, RED
0x0001000000000000 E

ATOM: mode
0x0000000000000000 ADD
0x0010000000000000 MIN
0x0020000000000000 MAX
0x0030000000000000 INC
0x0040000000000000 DEC
0x0050000000000000 AND
0x0060000000000000 OR
0x0070000000000000 XOR
0x0080000000000000 EXCH
0x03f0000000000000 CAS

ATOMS: type
0x0000000000000000
0x0000000010000000 .S32
0x0000000020000000 .U64
0x0000000030000000 .S64
0x0010000000000000 .64

ATOMS: mode
0x0000000000000000 ADD
0x0010000000000000 MIN
0x0020000000000000 MAX
0x0030000000000000 INC
0x0040000000000000 DEC
0x0050000000000000 AND
0x0060000000000000 OR
0x0070000000000000 XOR
0x0080000000000000 EXCH
0x0240000000000000 CAS
};

# The existence of a capture group can map directly to an op code adjustment, or...
# The named capture group value can map the op code adjustmemt from among several options
our %flags;
my (@ops, $flag);
foreach my $line (@flags)
{
    if ($line =~ m'^(0x[0-9a-z]+)\s*(.*)')
    {
        my $val = hex($1);
        # named rules (op: name)
        if ($flag)
            { $flags{$_}{$flag}{$2} = $val foreach @ops; }
        # simple existence check rules
        else
            { $flags{$_}{$2}        = $val foreach @ops; }
    }
    else
    {
        my ($ops, $name) = split ':\s*', $line;
        @ops = split ',\s*', $ops;
        $flag = $name;
    }
}

sub parseInstruct
{
    my ($inst, $grammar) = @_;
    return unless $inst =~ $grammar->{rule};
    my %capData = %+;
    return \%capData;
}

# for immediate or constant operands and a given opcode, bits 56-63 get transformed
my %immedOps = map { $_ => 1 } qw(i20 f20 d20);
my %immedCodes =
(
    0x5c => 0x64,
    0x5b => 0x6d,
    0x59 => 0x6b,
    0x58 => 0x68,
);
my %constCodes =
(
    c20 => 0x10,
    c39 => 0x08,
);
my %reuseCodes = (reuse1 => 1, reuse2 => 2, reuse3 => 4);

# just pick out the reuse code and nothing else
sub genReuseCode
{
    my $capData = shift;
    my $reuse = 0;
    $reuse |= $reuseCodes{$_} foreach grep $capData->{$_}, keys %reuseCodes;
    return $reuse;
}

# Generate an op code from regex capture data
# if you pass in a test array ref it will populate it with the matching capture groups
sub genCode
{
    my ($op, $grammar, $capData, $test) = @_;

    my $flags     = $flags{$op};
    my $code      = $grammar->{code};
    my $reuse     = 0;
    my $immedCode = $immedCodes{$code >> 56};

    #print map "$_: $capData->{$_}\n", keys %capData if $op eq 'I2I';

    # process the instruction predicate (if valid for this instuction)
    if (exists $capData->{noPred})
    {
        delete $capData->{noPred};
        push @$test, 'noPred' if $test;
    }
    else
    {
        my $p = defined($capData->{predNum}) ? $capData->{predNum} : 7;
        push @$test, 'predNum' if $test;
        if (exists $capData->{predNot})
        {
            $p |= 8;
            push @$test, 'predNot' if $test;
        }
        $code ^= $p << 16;
        delete @{$capData}{qw(predNum predNot)};

    }
    # process the register reuse flags
    foreach my $rcode (qw(reuse1 reuse2 reuse3))
    {
        if (delete $capData->{$rcode})
        {
            $reuse |= $reuseCodes{$rcode};
            push @$test, $rcode if $test;
        }
    }

    foreach my $capture (keys %$capData)
    {
        # change the base code for immediate versions of the op
        if (exists $immedOps{$capture})
            { $code ^= $immedCode << 56; }
        # change the base code for constant versions of the op
        elsif (exists $constCodes{$capture})
            { $code ^= $constCodes{$capture} << 56; }

        # if capture group is an operand then process and add that data to code
        if (exists $operands{$capture})
        {
            # don't process the r20 that comes with the r39s20 capture
            unless ($capture eq 'r20' && exists $capData->{r39s20})
            {
                $code ^= $operands{$capture}->($capData->{$capture});
                push @$test, $capture if $test;
            }
        }

        # Add matching flags (an operand might also add/remove a flag)
        if (exists $flags->{$capture})
        {
            # a named multivalue flag
            if (ref $flags->{$capture})
            {
                $code ^= $flags->{$capture}{$capData->{$capture}};
                push @$test, "$capture:$capData->{$capture}" if $test;
            }
            # a simple exists flag
            else
            {
                $code ^= $flags->{$capture};
                push @$test, $capture if $test;
            }
        }
        elsif (!exists $operands{$capture} && !$test)
        {
            # Every capture group should be acted upon.  Missing one is a bug.
            warn "UNUSED: $op: $capture: $capData->{$capture}\n";
            warn Dumper($flags);
        }
    }

    return $code, $reuse;
}


my $CtrlRe = qr'(?<ctrl>[0-9a-fA-F\-]{2}:[1-6\-]:[1-6\-]:[\-yY]:[0-9a-fA-F])';
my $PredRe = qr'(?<pred>@!?(?<predReg>P\d)\s+)';
my $InstRe = qr"$PredRe?(?<op>\w+)(?<rest>[^;]*;)"o;
my $CommRe = qr'(?<comment>.*)';

sub processAsmLine
{
    my ($line, $lineNum) = @_;

    if ($line =~ m"^$CtrlRe(?<space>\s+)$InstRe$CommRe"o)
    {
        return {
            lineNum => $lineNum,
            pred    => $+{pred},
            predReg => $+{predReg},
            space   => $+{space},
            op      => $+{op},
            comment => $+{comment},
            inst    => normalizeSpacing($+{pred} . $+{op} . $+{rest}),
            ctrl    => readCtrl($+{ctrl}, $line),
        };
    }
    return undef;
}

sub processSassLine
{
    my $line = shift;

    if ($line =~ m"^\s+/\*(?<num>[0-9a-f]+)\*/\s+$InstRe\s+/\* (?<code>0x[0-9a-f]+)"o)
    {
        return {
            num     => hex($+{num}),
            pred    => $+{pred},
            op      => $+{op},
            ins     => normalizeSpacing($+{op} . $+{rest}),
            inst    => normalizeSpacing($+{pred} . $+{op} . $+{rest}),
            code    => hex($+{code}),
        };
    }
    return undef;
}

sub processSassCtrlLine
{
    my ($line, $ctrl, $ruse) = @_;

    return 0 unless $line =~ m'^\s+\/\* (0x[0-9a-f]+)';

    my $code = hex($1);
    if (ref $ctrl)
    {
        push @$ctrl, ($code & 0x000000000001ffff) >> 0;
        push @$ctrl, ($code & 0x0000003fffe00000) >> 21;
        push @$ctrl, ($code & 0x07fffc0000000000) >> 42;
    }
    if (ref $ruse)
    {
        push @$ruse, ($code & 0x00000000001e0000) >> 17;
        push @$ruse, ($code & 0x000003c000000000) >> 38;
        push @$ruse, ($code & 0x7800000000000000) >> 59;
    }
    return 1;
}

sub replaceXMADs
{
    my $file = shift;

# XMAD.LO d, a, b, c, x;
# ----------------------
# XMAD.MRG x, a, b.H1, RZ;
# XMAD d, a, b, c;
# XMAD.PSL.CBCC d, a.H1, x.H1, d;
# ----------------------
# XMAD d, a, 0xffff, c;
# XMAD.PSL d, a.H1, 0xffff, d;
    $file =~ s/\n\s*$CtrlRe(?<space>\s+)($PredRe)?XMAD\.LO\s+(?<d>\w+)\s*,\s*(?<a>\w+)\s*,\s*(?<b>c\[$hex\]\[$hex\]|\w+)\s*,\s*(?<c>c\[$hex\]\[$hex\]|\w+)\s*,\s*(?<x>\w+)\s*;$CommRe/

        die "XMAD.LO: Destination and first operand cannot be the same register ($+{d})." if $+{d} eq $+{a};
        sprintf '
%1$s%2$s%3$sXMAD.MRG %8$s, %5$s, %6$s.H1, RZ;%9$s
%1$s%2$s%3$sXMAD %4$s, %5$s, %6$s, %7$s;
%1$s%2$s%3$sXMAD.PSL.CBCC %4$s, %5$s.H1, %8$s.H1, %4$s;',
                @+{qw(ctrl space pred d a b c x comment)}
    /egmos;

    $file =~ s/\n\s*$CtrlRe(?<space>\s+)($PredRe)?XMAD(?<mod>(?:\.[SU]16)(?:\.[SU]16))?\.LO2\s+(?<d>\w+)\s*,\s*(?<a>\w+)\s*,\s*(?<b>-?$immed|\w+)\s*,\s*(?<c>c\[$hex\]\[$hex\]|\w+)\s*;$CommRe/

        die "XMAD.LO2: Destination and first operand cannot be the same register ($+{d})." if $+{d} eq $+{a};
        sprintf '
%1$s%2$s%3$sXMAD%9$s %4$s, %5$s, %6$s, %7$s;%8$s
%1$s%2$s%3$sXMAD%9$s.PSL %4$s, %5$s.H1, %6$s, %4$s;',
            @+{qw(ctrl space pred d a b c comment mod)}
    /egmos;

    $file =~ s/\n\s*$CtrlRe(?<space>\s+)($PredRe)?XMAD(?<mod>(?:\.[SU]16)(?:\.[SU]16))?\.LO2C\s+(?<d>\w+)\s*,\s*(?<a>\w+)\s*,\s*(?<b>c\[$hex\]\[$hex\]|\w+)\s*,\s*(?<c>\w+)\s*;$CommRe/

        die "XMAD.LO2C: Destination and first operand cannot be the same register ($+{d})." if $+{d} eq $+{a};
        sprintf '
%1$s%2$s%3$sXMAD%9$s %4$s, %5$s, %6$s, %7$s;%8$s
%1$s%2$s%3$sXMAD%9$s.PSL %4$s, %5$s, %6$s.H1, %4$s;',
            @+{qw(ctrl space pred d a b c comment mod)}
    /egmos;

    #TODO: add more XMAD macros
    return $file;
}
# convert extra spaces to single spacing to make our re's simplier
sub normalizeSpacing
{
    my $inst = shift;
    $inst =~ s/\t/ /g;
    $inst =~ s/\s{2,}/ /g;
    return $inst;
}


# map binary control notation on to easier to work with format.
sub printCtrl
{
    my $code = shift;

    my $stall = ($code & 0x0000f) >> 0;
    my $yield = ($code & 0x00010) >> 4;
    my $wrtdb = ($code & 0x000e0) >> 5;  # write dependency barier
    my $readb = ($code & 0x00700) >> 8;  # read  dependency barier
    my $watdb = ($code & 0x1f800) >> 11; # wait on dependency barier

    $yield = $yield ? '-' : 'Y';
    $wrtdb = $wrtdb == 7 ? '-' : $wrtdb + 1;
    $readb = $readb == 7 ? '-' : $readb + 1;
    $watdb = $watdb ? sprintf('%02x', $watdb) : '--';

    return sprintf '%s:%s:%s:%s:%x', $watdb, $readb, $wrtdb, $yield, $stall;
}
sub readCtrl
{
    my ($ctrl, $context) = @_;
    my ($watdb, $readb, $wrtdb, $yield, $stall) = split ':', $ctrl;

    $watdb = $watdb eq '--' ? 0 : hex $watdb;
    $readb = $readb eq '-'  ? 7 : $readb - 1;
    $wrtdb = $wrtdb eq '-'  ? 7 : $wrtdb - 1;
    $yield = $yield eq 'y' || $yield eq 'Y'  ? 0 : 1;
    $stall = hex $stall;

    die sprintf('wait dep out of range(0x00-0x3f): %x at %s',   $watdb, $context) if $watdb != ($watdb & 0x3f);

    return
        $watdb << 11 |
        $readb << 8  |
        $wrtdb << 5  |
        $yield << 4  |
        $stall << 0;
}

sub getRegNum
{
    my ($regMap, $regName) = @_;

    return !exists($regMap->{$regName}) || ref($regMap->{$regName}) ? $regName : $regMap->{$regName};
}

sub getVecRegisters
{
    my ($vectors, $capData) = @_;
    my $regName = $capData->{r0} or return;

    return if $regName eq 'RZ';

    if ($capData->{type} eq '.64' || $capData->{i31w4} eq '0x3')
    {
        if ($regName =~ m'^R(\d+)$')
        {
            return map "R$_", ($1 .. $1+1);
        }
        confess "$regName not a 64bit vector register" unless exists $vectors->{$regName};
        return @{$vectors->{$regName}}[0,1];
    }
    if ($capData->{type} eq '.128' || $capData->{i31w4} eq '0xf')
    {
        if ($regName =~ m'^R(\d+)$')
        {
            return map "R$_", ($1 .. $1+3);
        }
        confess "$regName not a 128bit vector register" unless exists($vectors->{$regName}) && @{$vectors->{$regName}} == 4;
        return @{$vectors->{$regName}};
    }
    return $regName;
}

sub getAddrVecRegisters
{
    my ($vectors, $capData) = @_;
    my $regName = $capData->{r8} or return;

    return if $regName eq 'RZ';

    if (exists $capData->{E})
    {
        if ($regName =~ m'^R(\d+)$')
        {
            return map "R$_", ($1 .. $1+1);
        }
        print Dumper($vectors) unless exists $vectors->{$regName};
        confess "$regName not a 64bit vector register" unless exists $vectors->{$regName};
        return @{$vectors->{$regName}}[0,1];
    }
    return $regName;
}

__END__



