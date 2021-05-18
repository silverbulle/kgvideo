import h5py

f = h5py.File('MSR-VTT_BFeat.hdf5', 'r')
# f = h5py.File('MSR-VTT_I3D.hdf5', 'r')

for key in f.keys():
    # print(key)
    print(f[key].name)
    print(f[key].shape)
    # print(f[key].value)