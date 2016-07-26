import os
import tarfile
import sys
import csv
from subprocess import Popen, PIPE
from tqdm import tqdm
import numpy as np
from zipfile import ZipFile


def ingest_cifar10(out_dir, padded_size, overwrite=False):
    '''
    Save CIFAR-10 dataset as PNG files
    '''
    from neon.data import load_cifar10
    from PIL import Image
    dataset = dict()
    dataset['train'], dataset['val'], _ = load_cifar10(out_dir, normalize=False)
    pad_size = (padded_size - 32) // 2 if padded_size > 32 else 0
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))

    set_names = ('train', 'val')
    manifest_files = [os.path.join(out_dir, setn + '_file.csv') for setn in set_names]

    for setn, manifest in zip(set_names, manifest_files):
        data, labels = dataset[setn]

        img_dir = os.path.join(out_dir, setn)
        ulabels = np.unique(labels)
        for ulabel in ulabels:
            subdir = os.path.join(img_dir, str(ulabel))
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # write out label in a text file
            with open(os.path.join(img_dir, str(ulabel) + '.txt'), 'w') as f:
                f.write("%d" % ulabel)


        if (overwrite or not os.path.exists(manifest)):
            with open(manifest, 'w') as f:
                for idx in tqdm(range(data.shape[0])):
                    label = str(labels[idx][0])
                    im = np.pad(data[idx].reshape((3, 32, 32)), pad_width, mode='mean')
                    im = np.uint8(np.transpose(im, axes=[1, 2, 0]).copy())
                    im = Image.fromarray(im)
                    img_path = os.path.join(img_dir, label, str(idx) + '.png')
                    im.save(img_path, format='PNG')
                    f.write("{img_path},{lbl_path}\n".format(
                        img_path=img_path, lbl_path=os.path.join(img_dir, label + '.txt')))

    return manifest_files


def ingest_genre_data(in_tar, ingest_dir, train_percent=80):
    train_idx = os.path.join(ingest_dir, 'train-index.csv')
    val_idx = os.path.join(ingest_dir, 'val-index.csv')
    if os.path.exists(train_idx) and os.path.exists(val_idx):
        return train_idx, val_idx

    assert os.path.exists(in_tar)

    # convert files as we extract
    class_files = dict()
    with tarfile.open(in_tar, 'r') as tf_archive:
        infiles = [elem for elem in tf_archive.getmembers()]
        for tf_elem in tqdm(infiles):
            dirpath = tf_elem.name.split('/')
            outpath = os.path.join(ingest_dir, *dirpath[1:])
            if tf_elem.isdir() and not os.path.exists(outpath):
                os.makedirs(outpath)
            elif tf_elem.isfile():
                outfile = outpath.replace('.au', '.wav')
                class_files.setdefault(dirpath[1], []).append(outfile)
                a = tf_archive.extractfile(tf_elem)
                Popen(('sox', '-', outfile), stdin=PIPE).communicate(input=a.read())

    # make target files
    class_targets = dict()
    for label_idx, cls_name in enumerate(class_files.keys()):
        class_targets[cls_name] = os.path.join(ingest_dir, cls_name, cls_name + '.txt')
        with open(class_targets[cls_name], 'w') as label_fd:
            label_fd.write(str(label_idx) + '\n');

    np.random.seed(0)
    with open(train_idx, 'wb') as train_fd, open(val_idx, 'wb') as val_fd:
        train_csv, val_csv = csv.writer(train_fd), csv.writer(val_fd)
        for cls_name in class_files.keys():
            files, label = class_files[cls_name], class_targets[cls_name]
            np.random.shuffle(files)
            train_count = (len(files) * train_percent) // 100
            for filename in files[:train_count]:
                train_csv.writerow([filename, label])
            for filename in files[train_count:]:
                val_csv.writerow([filename, label])

    return train_idx, val_idx


def ingest_whale_data(in_zip, ingest_dir, train_percent=80):
    train_idx = os.path.join(ingest_dir, 'train-index.csv')
    val_idx = os.path.join(ingest_dir, 'val-index.csv')
    test_idx = os.path.join(ingest_dir, 'test-index.csv')
    all_idx = os.path.join(ingest_dir, 'all-index.csv')
    noise_idx = os.path.join(ingest_dir, 'noise-index.csv')

    if os.path.exists(train_idx) and os.path.exists(val_idx) and os.path.exists(test_idx):
        return train_idx, val_idx, test_idx, all_idx, noise_idx

    assert os.path.exists(in_zip)

    # convert files as we extract
    print("Extracting audio files from zip archive")
    class_files, test_files = dict(), []
    with ZipFile(in_zip, 'r') as zp_archive:
        with zp_archive.open('data/train.csv') as a:
            reader = csv.DictReader(a)
            for t in reader:
                aiff, lbl = t['clip_name'], t['label']
                outfile = os.path.join(ingest_dir, 'train', aiff.replace('.aiff', '.wav'))
                class_files.setdefault(lbl, []).append(outfile)

        for zp_elem in tqdm(zp_archive.infolist()):
            zpf = zp_elem.filename
            outpath = os.path.join(ingest_dir, *zpf.split('/')[1:]).replace('.aiff', '.wav')
            if zpf.endswith('/') and not os.path.exists(outpath):
                os.makedirs(outpath)
            elif zpf.endswith('.aiff'):
                austream = zp_archive.open(zp_elem)
                Popen(('sox', '-', outpath), stdin=PIPE).communicate(input=austream.read())
                if zpf.startswith('data/test'):
                    test_files.append(outpath)

    # make target files
    class_targets = dict()
    for labelidx in ['0', '1']:
        class_targets[labelidx] = os.path.join(ingest_dir, labelidx + '.txt')
        with open(class_targets[labelidx], 'w') as f:
            f.write(labelidx + '\n');

    np.random.seed(0)
    with open(train_idx, 'wb') as train_fd, open(val_idx, 'wb') as val_fd, \
     open(all_idx, 'wb') as all_fd, open(noise_idx, 'w') as noise_fd:
        train_csv, val_csv, all_csv = csv.writer(train_fd), csv.writer(val_fd), csv.writer(all_fd)
        for cls_name in class_files.keys():
            files, label = class_files[cls_name], class_targets[cls_name]
            np.random.shuffle(files)
            train_count = (len(files) * train_percent) // 100
            for filename in files[:train_count]:
                train_csv.writerow([filename, label])
                all_csv.writerow([filename, label])
                if cls_name == '0':
                    noise_fd.write(filename + '\n')
            for filename in files[train_count:]:
                val_csv.writerow([filename, label])
                all_csv.writerow([filename, label])

    with open(test_idx, 'w') as test_fd:
        for tf in test_files:
            test_fd.write(tf + '\n')

    return train_idx, val_idx, test_idx, all_idx, noise_idx

