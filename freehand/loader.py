
import random
import json

import h5py
import numpy as np


class SSFrameDataset():  # Subject-Scan frame loader

    def __init__(self, filename_h5, indices_in_use=None, num_samples=2, sample_range=None):

        """
        :param filename_h5, file path
        :param indices_in_use: 
            case 1: a list of tuples (idx_subject, idx_scans), indexing self.num_frames[indices_in_use[idx]]
            case 2: a list of two lists, [indices_subjects] and [indices_scans], meshgrid to get indices
            case 3: None (default), use all available in the file
        
        Sampling parameters
        :param num_samples: type int, number of (model input) frames, > 1. However, when num_samples=-1, sample all in the scan
        :param sample_range: type int, range of sampling frames, default is num_samples
        """
        
        self.filename = filename_h5
        self.file = h5py.File(self.filename, 'r')
        self.frame_size = self.file['frame_size'][()]
        self.num_frames = self.file['num_frames'][()]
        
        if indices_in_use is None:
            self.indices_in_use = [(i_sub,i_scn) for i_sub in range(self.num_frames.shape[0]) for i_scn in range(self.num_frames.shape[1])]                    
        elif all([isinstance(t,tuple) for t in indices_in_use]):
            self.indices_in_use = indices_in_use
        elif isinstance(indices_in_use[0],list) and isinstance(indices_in_use[1],list):
            self.indices_in_use = [(i_sub,i_scn) for i_sub in indices_in_use[0] for i_scn in indices_in_use[1]]            
        else:
            raise("indices_in_use should be a list of tuples (idx_subject, idx_scans) of two lists, [indices_subjects] and [indices_scans].")
        
        if len(set(self.indices_in_use)) != len(self.indices_in_use):
            print("WARNING: Replicated indices are found - not removed.")
        
        self.indices_in_use.sort()
        self.num_indices = len(self.indices_in_use)

        # sampling parameters
        if num_samples < 2:
            if num_samples == -1:
                if sample_range is not None:
                    sample_range = None
                    print("Sampling all frames. sample_range is ignored.")
            else:
                raise('num_samples should be greater than or equal to 2, or -1 for sampling all frames.')
        self.num_samples = num_samples
        
        if sample_range is None:
            self.sample_range = self.num_samples
        elif any([self.num_frames[indices]<sample_range for indices in self.indices_in_use]):
            raise("The specified sample_range is larger than number of frames in at least one of the in-use scans.")
        else:
            self.sample_range = sample_range
        
        

    def partition_by_ratio(self, ratios, randomise=False, subject_level=False):
        num_sets = len(ratios)
        ratios = [ratios[i]/sum(ratios) for i in range(num_sets)]
        print("Partitioning into %d sets with a normalised ratios %s," %
              (num_sets, ratios))

        if subject_level:
            raise('Subject-level split has not been implemented.')  # N.B. TBA

        else:  # scan-level split
            set_sizes = [int(self.num_indices*r) for r in ratios]
            for ii in range(self.num_indices-sum(set_sizes)): 
                set_sizes[ii]+=1  # add the remainders
            if randomise:
                random.shuffle(self.indices_in_use)            
            indices_sets = [self.indices_in_use[n0:n0+n1] for (n0, n1) in zip([sum(set_sizes[:ii]) for ii in range(num_sets)], set_sizes)]  # get the index tuples for all sets
            print("at scan-level, with %s scans." % (set_sizes))

            return [SSFrameDataset(filename_h5=self.filename, indices_in_use=idx_list, num_samples=self.num_samples, sample_range=self.sample_range) for idx_list in indices_sets]


    def __add__(self, other):
        if self.filename != other.filename:
            raise('Currently different file combining is not supported.')
        if self.num_samples != other.num_samples:
            print('WARNING: found different num_samples - the first is used.')
        if self.sample_range != other.sample_range:
            print('WARNING: found different sample_range - the first is used.')
        indices_combined = self.indices_in_use + other.indices_in_use
        return SSFrameDataset(filename_h5=self.filename, indices_in_use=indices_combined, num_samples=self.num_samples, sample_range=self.sample_range)
    

    def __len__(self):
        return self.num_indices
    

    def __getitem__(self, idx):
        indices = self.indices_in_use[idx]
        if self.num_samples == -1:  # sample all available frames, for validation
            i_frames = range(self.num_frames[indices])
        else:
            i_frames = self.frame_sampler(self.num_frames[indices])

        frames = [self.file['/sub{:03d}_scan{:02d}_frame{:04d}'.format(indices[0],indices[1],ii)] for ii in i_frames]
        tforms = [self.file['/sub{:03d}_scan{:02d}_tform{:04d}'.format(indices[0],indices[1],ii)] for ii in i_frames]
        tforms_inv = [self.file['/sub{:03d}_scan{:02d}_tform_inv{:04d}'.format(indices[0],indices[1],ii)] for ii in i_frames]

        return np.stack(frames,axis=0), np.stack(tforms,axis=0), np.stack(tforms_inv,axis=0)
    

    def frame_sampler(self, n):
        n0 = random.randint(0,n-self.sample_range)  # sample the start index for the range
        idx_frames = random.sample(range(n0,n0+self.sample_range), self.num_samples)   # sample indices
        idx_frames.sort()
        return idx_frames
    

    def write_json(self, jason_filename):
        with open(jason_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "filename" : self.filename, 
                "indices_in_use" : self.indices_in_use,
                "num_samples": self.num_samples,
                "sample_range": self.sample_range
                }, f, ensure_ascii=False, indent=4)
        print("%s written." % jason_filename)
    
    
    @staticmethod
    def read_json(jason_filename):
        with open(jason_filename, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            return SSFrameDataset(
                filename_h5    = obj['filename'], 
                indices_in_use = [tuple(ids) for ids in obj['indices_in_use']], # convert to tuples from json string
                num_samples    = obj['num_samples'], 
                sample_range   = obj['sample_range']
                )
        
            