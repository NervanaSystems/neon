#!/usr/bin/perl
use strict;
use MaxAs::Cubin;
use MaxAs::MaxAs;
use Data::Dumper;
use File::Spec;

require 5.10.0;

$Data::Dumper::Sortkeys = 1;

my $mode = shift;

# List cubin contents
if ($mode =~ /^\-?\-l/i)
{
    my $cubinFile = shift or usage();

    my $cubin = MaxAs::Cubin->new($cubinFile);

    my $arch    = $cubin->arch;
    my $class   = $cubin->class;
    my $asize   = $cubin->address_size;
    my $kernels = $cubin->listKernels;
    my $symbols = $cubin->listSymbols;

    printf "%s: arch:sm_%d machine:%dbit address_size:%dbit\n", $cubinFile, $arch, $class, $asize;

    foreach my $ker (sort keys %$kernels)
    {
        printf "Kernel: %s (Linkage: %s, Params: %d, Size: %d, Registers: %d, SharedMem: %d, Barriers: %d)\n", $ker, @{$kernels->{$ker}}{qw(Linkage ParamCnt size RegCnt SharedSize BarCnt)};
    }
    foreach my $sym (sort keys %$symbols)
    {
        printf "Symbol: %s\n", $sym;
    }
}
# Test that the assembler can reproduce the op codes this cubin or sass contains
elsif ($mode =~ /^\-?\-t/i)
{
    my $reg  = shift if $ARGV[0] =~ /^\-?\-r/i;
    my $all  = shift if $ARGV[0] =~ /^\-?\-a/i;
    my $file = shift or usage();
    my $fh;
    # sass file
    if (-T $file)
    {
        open $fh, $file or die "$file: $!";
    }
    # cubin file
    else
    {
        my $cubin = MaxAs::Cubin->new($file);
        my $arch  = $cubin->arch;

        open $fh, "cuobjdump -arch sm_$arch -sass $file |" or die "cuobjdump -arch sm_$arch -sass $file: $!";
        my $first = <$fh>;
        if ($first =~ /cuobjdump fatal/)
        {
            print $first;
            exit(1);
        }
    }
    exit(MaxAs::MaxAs::Test($fh, $reg, $all) ? 1 : 0);
}
# Extract an asm file containing the desired kernel
elsif ($mode =~ /^\-?\-e/i)
{
    my $kernelName;
    if ($ARGV[0] =~ /^\-?\-k/i)
    {
        shift;
        $kernelName = shift or usage();
    }
    my $cubinFile = shift or usage();
    my $asmFile   = shift;
    my $cubin     = MaxAs::Cubin->new($cubinFile);
    my $arch      = $cubin->arch;
    my $kernels   = $cubin->listKernels;

    #default the kernel name if not specified.
    $kernelName ||= (sort keys %$kernels)[0];

    my $kernel = $kernels->{$kernelName} or die "bad kernel: $kernelName";

    open my $in, "cuobjdump -arch sm_$arch -sass -fun $kernelName $cubinFile |" or die "cuobjdump -arch sm_50 -sass -fun $kernelName $cubinFile: $!";
    my $first = <$in>;
    if ($first =~ /cuobjdump fatal/)
    {
        print $first;
        exit(1);
    }
    my $out;
    if ($asmFile)
    {
        open $out, ">$asmFile" or die "$asmFile: $!";
    }
    else
    {
        $out = \*STDOUT;
    }

    print $out "# Kernel: $kernelName\n# Arch: sm_$arch\n";

    print $out "# $_: $kernel->{$_}\n" foreach (qw(InsCnt RegCnt SharedSize BarCnt));

    print $out "# Params($kernel->{ParamCnt}):\n#\tord:addr:size:align\n";

    print $out join('', map "#\t$_\n", @{$kernel->{Params}}) if $kernel->{Params};

    print $out "#\n# Instructions:\n\n";

    MaxAs::MaxAs::Extract($in, $out, $kernel->{Params});

    close $out if $asmFile;
    close $in;
}
# Extract a kernel from a sass dump
elsif ($mode =~ /^\-?\-s/i)
{
    my $sassFile  = shift or usage();
    my $asmFile   = shift;

    open my $in, $sassFile or die "$sassFile: $!";

    my $out;
    if ($asmFile)
    {
        open $out, ">$asmFile" or die "$asmFile: $!";
    }
    else
    {
        $out = \*STDOUT;
    }

    MaxAs::MaxAs::Extract($in, $out, []);

    close $out if $asmFile;
    close $in;
}
# Insert the kernel asm back into the cubin:
elsif ($mode =~ /^\-?\-i/i)
{
    my $nowarn;
    if ($ARGV[0] =~ /^\-?\-w/i)
    {
        $nowarn = shift;
    }
    my $kernelName;
    if ($ARGV[0] =~ /^\-?\-k/i)
    {
        shift;
        $kernelName = shift or usage();
    }
    my $noReuse   = shift if $ARGV[0] =~ /^\-?\-n/i;
    while ($ARGV[0] =~ /^\-?\-D(\w+)/)
    {
        shift;
        my $name  = $1;
        my $value = shift;
        eval "package MaxAs::MaxAs::CODE; our \$$name = '$value';"
    }

    my $asmFile   = shift or usage();
    my $cubinFile = shift or usage();
    my $newCubin  = shift || $cubinFile;

    my $file;
    if (open my $fh, $asmFile)
    {
        local $/;
        $file = <$fh>;
        close $fh;
    }
    else { die "$asmFile: $!" }

    my ($vol,$dir) = File::Spec->splitpath($asmFile);
    my $include = [$vol, $dir];

    # extract the kernel name from the file
    ($kernelName) = $file =~ /^# Kernel: (\w+)/ unless $kernelName;
    die "asm file missing kernel name or is badly formatted" unless $kernelName;

    my $kernel = MaxAs::MaxAs::Assemble($file, $include, !$noReuse, $nowarn);

    my $cubin  = MaxAs::Cubin->new($cubinFile);
    $kernel->{Kernel} = $cubin->getKernel($kernelName) or die "cubin does not contain kernel: $kernelName";

    $cubin->modifyKernel(%$kernel);

    $cubin->write($newCubin);

    printf "Kernel: $kernelName, Instructions: %d, Register Count: %d, Bank Conflicts: %d, Reuse: %.1f% (%d/%d)\n",
        @{$kernel}{qw(InsCnt RegCnt ConflictCnt ReusePct ReuseCnt ReuseTot)};

}
# Preprocessing:
elsif ($mode =~ /^\-?\-p/i)
{
    while ($ARGV[0] =~ /^\-?\-D(\w+)/)
    {
        shift;
        my $name  = $1;
        my $value = shift;
        eval "package MaxAs::MaxAs::CODE; our \$$name = '$value';";
    }
    my $debug     = shift if $ARGV[0] =~ /^\-?\-d/i;
    my $asmFile   = shift or usage();
    my $asmFile2  = shift;

    die "source and destination probably shouldn't be the same file\n" if $asmFile eq $asmFile2;

    open my $fh,  $asmFile or die "$asmFile: $!";
    local $/;
    my $file = <$fh>;
    close $fh;

    my ($vol,$dir) = File::Spec->splitpath($asmFile);
    my $include = [$vol, $dir];

    if ($asmFile2)
    {
        open $fh, ">$asmFile2" or die "$asmFile2: $!";
    }
    else
    {
        $fh = \*STDOUT;
    }
    print $fh MaxAs::MaxAs::Preprocess($file, $include, $debug);
    close $fh;
}
# get version information
elsif ($mode =~ /^\-?\-v/i)
{
    print "$MaxAs::MaxAs::VERSION\n";
}
else
{
    print "$mode\n";
    usage();
}

exit(0);



sub usage
{
    print <<EOF;
Usage:

  List kernels and symbols:

    maxas.pl --list|-l <cubin_file>

  Test a cubin or sass file to to see if the assembler can reproduce all of the contained opcodes.
  Also useful for extending the missing grammar rules.  Defaults to only showing failures without --all.
  With the --reg flag it will show register bank conflicts not hidden by reuse flags.

    maxas.pl --test|-t [--reg|-r] [--all|-a] <cubin_file | cuobjdump_sass_file>

  Extract a single kernel into an asm file from a cubin.
  Works much like cuobjdump but outputs in a format that can be re-assembled back into the cubin.

    maxas.pl --extract|-e [--kernel|-k kernel_name] <cubin_file> [asm_file]

  Preprocess the asm: expand CODE sections, perform scheduling. Mainly used for debugging purposes.
  Include the debug flag to print out detailed scheduler info.

    maxas.pl --pre|-p [--debug|-d] <asm_file> [new_asm_file]

  Insert the kernel asm back into the cubin.  Overwrite existing or create new cubin.
  Optionally you can skip register reuse flag auto insertion.  This allows you to observe
  performance without any reuse or you can use it to set the flags manually in your sass.

    maxas.pl --insert|-i [--noreuse|-n] <asm_file> <cubin_file> [new_cubin_file]

  Display version information and exit:

    maxas.pl --version|-v

EOF
    exit(1);
}

__END__
