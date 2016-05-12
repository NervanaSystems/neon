package MaxAs::Cubin;

use strict;
use Data::Dumper;

my @Elf32_Hdr = qw(
    H8  magic
    C   fileClass
    C   encoding
    C   fileVersion
    H18 padding
    S   type
    S   machine
    L   version
    L   entry
    L   phOffset
    L   shOffset
    L   flags
    S   ehSize
    S   phEntSize
    S   phNum
    S   shEntSize
    S   shNum
    S   shStrIndx
);
my @Elf64_Hdr = qw(
    H8  magic
    C   fileClass
    C   encoding
    C   fileVersion
    H18 padding
    S   type
    S   machine
    L   version
    Q   entry
    Q   phOffset
    Q   shOffset
    L   flags
    S   ehSize
    S   phEntSize
    S   phNum
    S   shEntSize
    S   shNum
    S   shStrIndx
);
my @Elf32_PrgHdr = qw(
    L   type
    L   offset
    L   vaddr
    L   paddr
    L   fileSize
    L   memSize
    L   flags
    L   align
);
my @Elf64_PrgHdr = qw(
    L   type
    L   flags
    Q   offset
    Q   vaddr
    Q   paddr
    Q   fileSize
    Q   memSize
    Q   align
);
my @Elf32_SecHdr = qw(
    L   name
    L   type
    L   flags
    L   addr
    L   offset
    L   size
    L   link
    L   info
    L   align
    L   entSize
);
my @Elf64_SecHdr = qw(
    L   name
    L   type
    Q   flags
    Q   addr
    Q   offset
    Q   size
    L   link
    L   info
    Q   align
    Q   entSize
);
my @Elf32_SymEnt = qw(
    L   name
    L   value
    L   size
    C   info
    C   other
    S   shIndx
);
my @Elf64_SymEnt = qw(
    L   name
    C   info
    C   other
    S   shIndx
    Q   value
    Q   size
);
my @symBind = qw(LOCAL GLOBAL WEAK);

# Split the Elf Header defs into template strings (T) and corresponding hash keys columns (C)
my (@elfHdrT, @prgHdrT, @secHdrT, @symHdrT, @elfHdrC, @prgHdrC, @secHdrC, @symHdrC);

$elfHdrT[1] = join '', grep { length($_) <= 3} @Elf32_Hdr;
$prgHdrT[1] = join '', grep { length($_) <= 3} @Elf32_PrgHdr;
$secHdrT[1] = join '', grep { length($_) <= 3} @Elf32_SecHdr;
$symHdrT[1] = join '', grep { length($_) <= 3} @Elf32_SymEnt;

$elfHdrT[2] = join '', grep { length($_) <= 3} @Elf64_Hdr;
$prgHdrT[2] = join '', grep { length($_) <= 3} @Elf64_PrgHdr;
$secHdrT[2] = join '', grep { length($_) <= 3} @Elf64_SecHdr;
$symHdrT[2] = join '', grep { length($_) <= 3} @Elf64_SymEnt;

$elfHdrC[1] = [ grep { length($_) > 3} @Elf32_Hdr    ];
$prgHdrC[1] = [ grep { length($_) > 3} @Elf32_PrgHdr ];
$secHdrC[1] = [ grep { length($_) > 3} @Elf32_SecHdr ];
$symHdrC[1] = [ grep { length($_) > 3} @Elf32_SymEnt ];

$elfHdrC[2] = [ grep { length($_) > 3} @Elf64_Hdr    ];
$prgHdrC[2] = [ grep { length($_) > 3} @Elf64_PrgHdr ];
$secHdrC[2] = [ grep { length($_) > 3} @Elf64_SecHdr ];
$symHdrC[2] = [ grep { length($_) > 3} @Elf64_SymEnt ];

# Load a cubin ELF file
sub new
{
    my ($package, $file) = @_;

    my $cubin = bless { fileName => $file }, $package;

    open my $fh, $file or die "$file: $!";
    binmode($fh);

    # Read in assuming 32 bit header
    my $data;
    read $fh, $data, 0x34;
    my $elfHdr = $cubin->{elfHdr} = {};
    @{$elfHdr}{@{$elfHdrC[1]}} = unpack $elfHdrT[1], $data;

    # 1: 32bit, 2: 64bit
    my $class = $elfHdr->{fileClass};

    # re-read in with 64 bit header if needed
    if ($class == 2)
    {
        seek $fh, 0, 0;
        read $fh, $data, 0x46;
        @{$elfHdr}{@{$elfHdrC[$class]}} = unpack $elfHdrT[$class], $data;

        $cubin->{Class} = 64;
    }
    else
    {
        $cubin->{Class} = 32;
    }

    # verify sm_50 cubin
    $cubin->{Arch} = $elfHdr->{flags} & 0xFF;
    die "Cubin not in sm_50 or greater format. Found: sm_$cubin->{Arch}\n" if $cubin->{Arch} < 50;

    $cubin->{AddressSize} = $elfHdr->{flags} & 0x400 ? 64 : 32;

    # Read in Program Headers
    seek $fh, $elfHdr->{phOffset}, 0;
    foreach (1 .. $elfHdr->{phNum})
    {
        read $fh, $data, $elfHdr->{phEntSize};

        my %prgHdr = (Indx => $_ - 1);
        @prgHdr{@{$prgHdrC[$class]}} = unpack $prgHdrT[$class], $data;
        push @{$cubin->{prgHdrs}}, \%prgHdr;
    }

    # Read in Section Headers
    seek $fh, $elfHdr->{shOffset}, 0;
    foreach (1 .. $elfHdr->{shNum})
    {
        read $fh, $data, $elfHdr->{shEntSize};

        my %secHdr = (Indx => $_ - 1);
        @secHdr{@{$secHdrC[$class]}} = unpack $secHdrT[$class], $data;
        push @{$cubin->{secHdrs}}, \%secHdr;
    }

    # Read in Section data
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        $data = '';
        # Skip sections with no data (type NULL or NOBITS)
        if ($secHdr->{size} && $secHdr->{type} != 8)
        {
            seek $fh, $secHdr->{offset}, 0;
            read $fh, $data, $secHdr->{size};
        }
        # Convert string tables to maps
        if ($secHdr->{type} == 3) # STRTAB
        {
            my $strTab = $secHdr->{StrTab} = {};
            my $indx   = 0;
            foreach my $str (split "\0", $data)
            {
                $strTab->{$indx} = $str;
                $indx += 1 + length($str);
            }
        }
        # Read in Symbol data
        if ($secHdr->{type} == 2) # SYMTAB
        {
            my $offset = 0;
            while ($offset < $secHdr->{size})
            {
                my $symEnt = {};
                @{$symEnt}{@{$symHdrC[$class]}} = unpack $symHdrT[$class], substr($data, $offset, $secHdr->{entSize});
                $offset += $secHdr->{entSize};

                push @{$secHdr->{SymTab}}, $symEnt;
            }
        }
        # Cache raw data for further processing and writing
        $secHdr->{Data} = unpack 'H*', $data;
    }
    close $fh;

    # Update section headers with their names.  Map names directly to headers.
    my $shStrTab = $cubin->{secHdrs}[$elfHdr->{shStrIndx}]{StrTab};
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        $secHdr->{Name} = $shStrTab->{$secHdr->{name}};
        $cubin->{$secHdr->{Name}} = $secHdr;
    }

    # Update symbols with their names
    # For the Global functions, extract kernel meta data
    # Populate the kernel hash
    my $strTab = $cubin->{'.strtab'}{StrTab};
    foreach my $symEnt (@{$cubin->{'.symtab'}{SymTab}})
    {
        $symEnt->{Name} = $strTab->{$symEnt->{name}};

        # Attach symbol to section
        my $secHdr = $cubin->{secHdrs}[$symEnt->{shIndx}];
        $secHdr->{SymbolEnt} = $symEnt;

        # Look for symbols tagged FUNC
        if (($symEnt->{info} & 0x0f) == 0x02)
        {
            # Create a hash of kernels for output
            my $kernelSec = $cubin->{Kernels}{$symEnt->{Name}} = $secHdr;

            # Extract local/global/weak binding info
            $kernelSec->{Linkage} = $symBind[($symEnt->{info} & 0xf0) >> 4];

            # Extract the kernel instructions
            $kernelSec->{KernelData} = [ unpack "Q*", pack "H*", $kernelSec->{Data} ];

            # Extract the max barrier resource identifier used and add 1. Should be 0-16.
            # If a register is used as a barrier resource id, then this value is the max of 16.
            $kernelSec->{BarCnt} = ($kernelSec->{flags} & 0x01f00000) >> 20;

            # Extract the number of allocated registers for this kernel.
            $kernelSec->{RegCnt} = ($kernelSec->{info} & 0xff000000) >> 24;

            # Extract the size of shared memory this kernel uses.
            my $sharedSec = $kernelSec->{SharedSec} = $cubin->{".nv.shared.$symEnt->{Name}"};
            $kernelSec->{SharedSize} = $sharedSec ? $sharedSec->{size} : 0;

            # Attach constant0 section
            $kernelSec->{ConstantSec} = $cubin->{".nv.constant0.$symEnt->{Name}"};

            # Extract the kernel parameter data.
            my $paramSec = $kernelSec->{ParamSec} = $cubin->{".nv.info.$symEnt->{Name}"};
            if ($paramSec)
            {
                # Extract raw param data
                my @data = unpack "L*", pack "H*", $paramSec->{Data};

                $paramSec->{ParamData} = \@data;
                $paramSec->{ParamHex} = [ map { sprintf '0x%08x', $_ } @data ];

                # Find the first param delimiter
                my $idx = 0;
                $idx++ while $idx < @data && $data[$idx] != 0x00080a04;

                my $first = $data[$idx+2] & 0xFFFF;
                #my $size  = $data[$idx+2] >> 16;
                $idx += 4;

                my @params;
                while ($idx < @data && $data[$idx] == 0x000c1704)
                {
                    # Get the ordinal, offset, size and pointer alignment for each param
                    my $ord    = $data[$idx+2] & 0xFFFF;
                    my $offset = sprintf '0x%02x', $first + ($data[$idx+2] >> 16);
                    my $psize  = $data[$idx+3] >> 18;
                    my $align  = $data[$idx+3] & 0x400 ? 1 << ($data[$idx+3] & 0x3ff) : 0;
                    unshift @params, "$ord:$offset:$psize:$align";
                    $idx += 4;
                }
                my @staticParams = @data[0 .. ($idx-1)];

                my ($maxregCount, @exitOffsets, @ctaidOffsets, $ctaidzUsed, @reqntid, @maxntid, @stackSize);
                while ($idx < @data)
                {
                    my $code = $data[$idx] & 0xffff;
                    my $size = $data[$idx] >> 16;
                    $idx++;

                    # EIATTR_MAXREG_COUNT
                    if ($code == 0x1b03)
                    {
                        $maxregCount = $size;
                    }
                    # EIATTR_S2RCTAID_INSTR_OFFSETS
                    elsif ($code == 0x1d04)
                    {
                        while ($size > 0)
                        {
                            push @ctaidOffsets, $data[$idx++];
                            $size -= 4;
                        }
                    }
                    # EIATTR_EXIT_INSTR_OFFSETS
                    elsif ($code == 0x1c04)
                    {
                        while ($size > 0)
                        {
                            push @exitOffsets, $data[$idx++];
                            $size -= 4;
                        }
                    }
                    # EIATTR_CTAIDZ_USED
                    elsif ($code == 0x0401)
                    {
                        $ctaidzUsed = 1;
                    }
                    # EIATTR_REQNTID
                    elsif ($code == 0x1004)
                    {
                        while ($size > 0)
                        {
                            push @reqntid, $data[$idx++];
                            $size -= 4;
                        }
                    }
                    # EIATTR_MAX_THREADS
                    elsif ($code == 0x0504)
                    {
                        while ($size > 0)
                        {
                            push @maxntid, $data[$idx++];
                            $size -= 4;
                        }
                    }
                    # EIATTR_CRS_STACK_SIZE
                    elsif ($code == 0x1e04)
                    {
                        while ($size > 0)
                        {
                            push @stackSize, $data[$idx++];
                            $size -= 4;
                        }
                    }
                    else
                    {
                        printf STDERR "Unknown Code 0x%02x (size:%d)\n", $code, $size;
                    }
                }
                $kernelSec->{Params}   = \@params;
                $kernelSec->{ParamCnt} = scalar @params;

                $paramSec->{StaticParams} = \@staticParams;
                $paramSec->{MAXREG_COUNT} = $maxregCount;
                $paramSec->{ExitOffsets}  = \@exitOffsets;
                $paramSec->{CTAIDOffsets} = \@ctaidOffsets;
                $paramSec->{CTAIDZUsed}   = $ctaidzUsed;
                $paramSec->{REQNTID}      = \@reqntid;
                $paramSec->{MAXNTID}      = \@maxntid;
                $paramSec->{STACKSIZE}    = \@stackSize;
            }
            # print Dumper($paramSec);
            # exit();
        }
        # Note GLOBALs found in this cubin
        elsif (($symEnt->{info} & 0x10) == 0x10)
        {
            $cubin->{Symbols}{$symEnt->{Name}} = $symEnt;
        }
    }

    # print "phOffset: $elfHdr->{phOffset}\n";
    # print "shOffset: $elfHdr->{shOffset}\n";
    # foreach my $secHdr (@{$cubin->{secHdrs}})
    # {
    #     print "secHdr($secHdr->{Indx}): $secHdr->{offset}, $secHdr->{size}, $secHdr->{align} ($secHdr->{Name})\n";
    # }
    # my $p = 0;
    # foreach my $prgHdr (@{$cubin->{prgHdrs}})
    # {
    #     print "prgHdr($p): type: $prgHdr->{type}, offset: $prgHdr->{offset}, fileSize: $prgHdr->{fileSize}, memSize: $prgHdr->{memSize}, align: $prgHdr->{align}\n";
    #     $p++;
    # }
    # exit();

    # print Dumper($cubin->{prgHdrs});
    # exit();
    return $cubin;
}
sub class
{
    return shift()->{Class};
}
sub arch
{
    return shift()->{Arch};
}
sub address_size
{
    return shift()->{AddressSize};
}
sub listKernels
{
    return shift()->{Kernels};
}
sub listSymbols
{
    return shift()->{Symbols};
}
sub getKernel
{
    my ($cubin, $kernel) = @_;
    return $cubin->{Kernels}{$kernel};
}

sub modifyKernel
{
    my ($cubin, %params) = @_;

    my $kernelSec    = $params{Kernel};
    my $newReg       = $params{RegCnt};
    my $newBar       = $params{BarCnt};
    my $exitOffsets  = $params{ExitOffsets};
    my $ctaidOffsets = $params{CTAIDOffsets};
    my $ctaidzUsed   = $params{CTAIDZUsed};
    my $newData      = $params{KernelData};
    my $newSize      = @$newData * 8;

    die "255 register max" if $newReg > 255;
    die "new kernel size must be multiple of 8 instructions (64 bytes)" if $newSize & 63;
    die "16 is max barrier count" if $newBar > 16;

    my $paramSec    = $kernelSec->{ParamSec};
    my $kernelName  = $kernelSec->{SymbolEnt}{Name};
    my $maxregCount = $paramSec->{MAXREG_COUNT};
    my $stackSize   = $paramSec->{STACKSIZE};

    # update the kernel
    $kernelSec->{KernelData} = $newData;
    $kernelSec->{Data}       = unpack "H*", pack "Q*", @$newData;

    if ($newReg != $kernelSec->{RegCnt})
    {
        print "Modified $kernelName RegCnt: $kernelSec->{RegCnt} => $newReg\n";
        $kernelSec->{RegCnt} = $newReg;
        $kernelSec->{info}  &= ~0xff000000;
        $kernelSec->{info}  |= $newReg << 24;
    }
    if ($newBar != $kernelSec->{BarCnt})
    {
        print "Modified $kernelName BarCnt: $kernelSec->{BarCnt} => $newBar\n";
        $kernelSec->{BarCnt} = $newBar;
        $kernelSec->{flags} &= ~0x01f00000;
        $kernelSec->{flags} |=  $newBar << 20;
    }

    my @paramData = @{$paramSec->{StaticParams}};

    if (defined $maxregCount)
    {
        push @paramData, ($maxregCount << 16) | 0x1b03;
    }

    my $newCTAIDs = join ',', map { sprintf '%04x', $_ } @$ctaidOffsets;
    my $oldCTAIDs = join ',', map { sprintf '%04x', $_ } @{$paramSec->{CTAIDOffsets}};

    if ($newCTAIDs ne $oldCTAIDs)
    {
        print "Modified $kernelName CTAID Offsets: '$oldCTAIDs' => '$newCTAIDs'\n";
    }
    if (@$ctaidOffsets)
    {
        push @paramData, (scalar(@$ctaidOffsets) << 18) | 0x1d04;
        push @paramData, @$ctaidOffsets;
    }

    my $newExits = join ',', map { sprintf '%04x', $_ } @$exitOffsets;
    my $oldExits = join ',', map { sprintf '%04x', $_ } @{$paramSec->{ExitOffsets}};

    if ($newExits ne $oldExits)
    {
        print "Modified $kernelName Exit Offsets: '$oldExits' => '$newExits'\n";
    }
    if (@$exitOffsets)
    {
        push @paramData, (scalar(@$exitOffsets) << 18) | 0x1c04;
        push @paramData, @$exitOffsets;
    }

    if ($ctaidzUsed != $paramSec->{CTAIDZUsed})
    {
        print "Modified $kernelName CTAID.Z Used: '$paramSec->{CTAIDZUsed}' => '$ctaidzUsed'\n";
    }
    if ($ctaidzUsed)
    {
        push @paramData, 0x0401;
    }

    if (@{$paramSec->{REQNTID}})
    {
        push @paramData, (scalar(@{$paramSec->{REQNTID}}) << 18) | 0x1004;
        push @paramData, @{$paramSec->{REQNTID}};
    }
    if (@{$paramSec->{MAXNTID}})
    {
        push @paramData, (scalar(@{$paramSec->{MAXNTID}}) << 18) | 0x0504;
        push @paramData, @{$paramSec->{MAXNTID}};
    }

    if (@$stackSize)
    {
        push @paramData, (scalar(@$stackSize) << 18) | 0x1e04;
        push @paramData, @$stackSize;
    }

    my $newParamSize  = scalar(@paramData)*4;
    $paramSec->{Data} = unpack "H*", pack "L*", @paramData;
    if ($newParamSize != $paramSec->{size})
    {
        print "Modified $kernelName ParamSecSize: $paramSec->{size} => $newParamSize\n";
        $cubin->updateSize($paramSec, $newParamSize);
    }

    if ($newSize != $kernelSec->{size})
    {
        print "Modified $kernelName KernelSize: $kernelSec->{size} => $newSize\n";
        $cubin->updateSize($kernelSec, $newSize, 1);
    }
}

sub updateSize
{
    my ($cubin, $sec, $newSize, $updatePrgSize) = @_;

    my $elfHdr = $cubin->{elfHdr};
    my $class  = $elfHdr->{fileClass};

    # update section header
    my $delta = $newSize - $sec->{size};
    $sec->{size} = $newSize;

    # update symtab section
    if ($sec->{SymbolEnt})
    {
        $sec->{SymbolEnt}{size} = $newSize;
        my $symSection = $cubin->{'.symtab'};
        $symSection->{Data} = '';
        foreach my $symEnt (@{$symSection->{SymTab}})
        {
            $symSection->{Data} .= unpack "H*", pack $symHdrT[$class], @{$symEnt}{@{$symHdrC[$class]}};
        }
    }

    my $pos = $elfHdr->{ehSize};
    my %sizeMap;

    # update section header offsets
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        # skip first header
        next if $secHdr->{align} == 0;

        # NOBITS data sections are size 0
        my $size = $secHdr->{type} == 8 ? 0 : $secHdr->{size};

        # Add any needed padding between sections
        my $pad = $pos % $secHdr->{align};
        if ($pad > 0)
        {
            $pos += $secHdr->{align} - $pad;
        }
        # map old offset to new
        $sizeMap{$secHdr->{offset}} = $pos;

        # update offset
        $secHdr->{offset} = $pos;

        # advance position by size
        $pos += $size;
    }

    # compute total section header size
    my $shSize = $elfHdr->{phOffset} - $elfHdr->{shOffset};

    # map old offset to new
    $sizeMap{$elfHdr->{shOffset}} = $pos;
    $sizeMap{$elfHdr->{phOffset}} = $pos + $shSize;

    $elfHdr->{shOffset} = $pos;
    $elfHdr->{phOffset} = $pos + $shSize;

    # update program header offsets and sizes
    foreach my $prgHdr (@{$cubin->{prgHdrs}})
    {
        # Not sure how best to adjust these so just assume they'll track other offsets.
        $prgHdr->{offset} = $sizeMap{$prgHdr->{offset}};

        # If the kernel sizes changes, also update the associated ProgramHeader.
        # Note that this size is the kernel size plus any constant section sizes.
        if ($updatePrgSize && $prgHdr->{type} == 1 &&
            $sec->{offset} >= $prgHdr->{offset} &&
            $sec->{offset} < $prgHdr->{offset} + $prgHdr->{fileSize} + $delta)
        {
            $prgHdr->{fileSize} += $delta;
            $prgHdr->{memSize}  += $delta;
        }
    }
}

# Write out the cubin after modifying it.
sub write
{
    my ($cubin, $file) = @_;

    open my $fh, ">$file" or die "Error: could not open $file for writing: $!";
    binmode($fh);

    my $elfHdr = $cubin->{elfHdr};
    my $class  = $elfHdr->{fileClass};

    # write elf header
    print $fh pack $elfHdrT[$class], @{$elfHdr}{@{$elfHdrC[$class]}};
    my $pos = $elfHdr->{ehSize};

    # write section data
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        # Skip NULL and NOBITS data sections
        next if $secHdr->{size} == 0 || $secHdr->{type} == 8;

        # Add any needed padding between sections
        my $pad = $pos % $secHdr->{align};
        if ($pad > 0)
        {
            $pad = $secHdr->{align} - $pad;
            print $fh join '', "\0" x $pad;
            $pos += $pad;
        }

        print $fh pack 'H*', $secHdr->{Data};
        $pos += $secHdr->{size};
    }

    # write section headers
    foreach my $secHdr (@{$cubin->{secHdrs}})
    {
        print $fh pack $secHdrT[$class], @{$secHdr}{@{$secHdrC[$class]}};
    }

    #write program headers
    foreach my $prgHdr (@{$cubin->{prgHdrs}})
    {
        print $fh pack $prgHdrT[$class], @{$prgHdr}{@{$prgHdrC[$class]}};
    }
    close $fh;
}

__END__

