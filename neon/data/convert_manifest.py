import os
import shutil
"""

In neon v2.2, we moved to using a new version of aeon (v1.0+). This version
uses a slightly different manifest format. Here we have provided a helper
script to convert manifest to his new format. Importantly, we use tab spacing,
and also include a header.
"""


def convert_manifest(source_manifest, output_manifest):
    """
    Converts manifest created for previous aeon versions for
    use with aeon v1.0+.

    Args:
         source_manifest: Path to old manifest
         output_manifest: Path to save converted manifest.
    """
    tmp_manifest = '/tmp/manifest{}'.format(os.getpid())
    tmp_dest = open(tmp_manifest, 'w')
    source = open(source_manifest, 'r')
    record = source.readline()
    splited = record.split(',')
    headerbody = "FILE\t" * len(splited)
    header = "@" + headerbody[:-1] + '\n'
    tmp_dest.write(header)
    record = record.replace(',', '\t')
    tmp_dest.write(record)

    for record in source:
        record = record.replace(',', '\t')
        tmp_dest.write(record)

    source.close()
    tmp_dest.close()

    if output_manifest is None:
        output_manifest = source_manifest

    if os.path.exists(output_manifest):
        os.remove(output_manifest)
    shutil.move(tmp_manifest, output_manifest)


if __name__ == '__main__':
    from configargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--manifest_file', required=True, help='Manifest to convert')
    parser.add_argument('--destination', help='Converted Manifest destination')
    args = parser.parse_args()

    convert_manifest(args.manifest_file, args.destination)
