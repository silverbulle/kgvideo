import h5py

f = h5py.File('MSR-VTT_OFeat_train.hdf5', 'r')
# f = h5py.File('MSR-VTT_OFeat1.hdf5', 'r')
# f = h5py.File('MSR-VTT_I3D.hdf5', 'r')
# load_video = h5py.File('msrvtt_roi_box.h5', 'r')
# f = h5py.File('msrvtt_roi_feat.h5', 'r')
# save_video = h5py.File('MSR-VTT_OFeat1.hdf5', 'w')
print(f['video999'].value)
print(f['video999'].shape)
breakpoint()
for vid in f.keys():
    # vid1 = 'video' + vid
    # save_video[vid1] = f[vid].value
    #
    # print(vid1 + ' have saved!!!')
    print(f[vid].name)
    print(f[vid].shape)
    print(f[vid].value)
# save_video.close()