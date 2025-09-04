import os
import urllib.request
import gzip, shutil
import hashlib
import tables
import numpy as np
from tools.analysis.signals import SpikeList

def retrieve_dataset(base_url, data_path):
    cache_dir = os.path.expanduser(data_path)
    cache_subdir = "hdspikes"
    print("Using cache dir: %s" % cache_dir)

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen("%s/md5sums.txt" % base_url)
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}

    def download_file(url, filepath, md5hash=None):
        """Download file with MD5 verification, similar to tf.keras.utils.get_file"""
        if os.path.exists(filepath):
            if md5hash:
                # Verify existing file
                with open(filepath, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash == md5hash:
                    return filepath
                else:
                    print(f"MD5 mismatch for {filepath}, re-downloading...")
            else:
                return filepath
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        print(f"Downloading {url} to {filepath}")
        urllib.request.urlretrieve(url, filepath)
        
        # Verify MD5 if provided
        if md5hash:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash != md5hash:
                os.remove(filepath)
                raise ValueError(f"MD5 hash mismatch for {filepath}")
        
        return filepath

    def get_and_gunzip(origin, filename, md5hash=None):
        cache_path = os.path.join(cache_dir, cache_subdir)
        gz_file_path = os.path.join(cache_path, filename)
        gz_file_path = download_file(origin, gz_file_path, md5hash)
        
        hdf5_file_path = gz_file_path[:-3]
        if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
            print("Decompressing %s" % gz_file_path)
            with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return hdf5_file_path

    # Download the Spiking Heidelberg Digits (SHD) dataset (the train subset)
    origin = "%s/%s" % (base_url, "shd_train.h5.gz")
    hdf5_file_path = get_and_gunzip(origin, "shd_train.h5.gz", md5hash=file_hashes["shd_train.h5.gz"])
    return hdf5_file_path


def retrieve_spike_lists(hdf5_file_path, n_samples, n_units):
    # At this point we can visualize some of the data
    fileh = tables.open_file(hdf5_file_path, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels

    sample_ids = np.arange(0, len(labels))
    np.random.shuffle(sample_ids)
    sample_ids = sample_ids[:n_samples]
    time_offset = 0.
    samples = []
    for sample in range(n_samples):
        tmp_spikes = [(units[sample][idx], np.round(x * 1000, 1)) for idx, x in enumerate(times[sample])]
        sl = SpikeList(tmp_spikes, list(np.unique(units[sample])))
        neuron_ids = np.array(sl.id_list)
        np.random.shuffle(neuron_ids)
        sl = sl.id_slice(list(neuron_ids[:n_units]), re_number=True)
        sl.time_offset(time_offset)
        time_offset += sl.t_stop
        samples.append((labels[sample], (sl.t_start, sl.t_stop), sl))
    return samples