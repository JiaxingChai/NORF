from datasets.sun_loader import *

statistics = torch.load("/ssd/cjx/semi-supervised/VPS/lib/dataloader/statistics.pth")     # "/ssd/cjx/semi-supervised/VPS/lib/dataloader/statistics.pth"
mean = statistics['mean']
std = statistics['std']

def get_sun_video_dataset(img_size, train_frames_per_clip, type='normal'):
    train_trsf_main = Compose_imglabel([
        Resize_video(img_size, img_size),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(mean, std)
    ])
    test_trsf_main = Compose_imglabel([
        Resize_video(img_size, img_size),
        toTensor_video(),
        Normalize_video(mean, std)
    ])
    
    train_ds = TrainDataset(transform=train_trsf_main, video_time_clips=train_frames_per_clip, type=type)
    test_ds = ValDataset(transform=test_trsf_main, video_time_clips=train_frames_per_clip, type=type)

    return train_ds, test_ds


def get_sun_video_datasetV2(img_size, train_frames_per_clip, memory_size=3, type='normal'):
    train_trsf_main = Compose_imglabel([
        Resize_video(img_size, img_size),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(mean, std),
    ])
    test_trsf_main = Compose_imglabel([
        Resize_video(img_size, img_size),
        toTensor_video(),
        Normalize_video(mean, std),
    ])
    
    train_ds = TrainDatasetV2(transform=train_trsf_main, video_time_clips=train_frames_per_clip, memory_size=memory_size, type=type)
    test_ds = ValDatasetV2(transform=test_trsf_main, video_time_clips=train_frames_per_clip, memory_size=memory_size, type=type)

    return train_ds, test_ds


def get_sun_pesudo_video_dataset(img_size, train_frames_per_clip, type='normal'):
    test_trsf_main = Compose_imglabel([
        Resize_video(img_size, img_size),
        toTensor_video(),
        Normalize_video(mean, std)
    ])
    train_ds = SUN_PesudoDataset(img_size=img_size,
                                 video_time_clips=train_frames_per_clip,
                                 type=type)
    test_ds = ValDataset(transform=test_trsf_main, video_time_clips=train_frames_per_clip, type=type)
    
    return train_ds, test_ds


def get_sun_pesudo_video_datasetV2(img_size, train_frames_per_clip, memory_size, type='normal'):
    test_trsf_main = Compose_imglabel([
        Resize_video(img_size, img_size),
        toTensor_video(),
        Normalize_video(mean, std)
    ])
    train_ds = SUN_PesudoDatasetV2(img_size=img_size, video_time_clips=train_frames_per_clip, memory_size=memory_size, type=type)
    test_ds = ValDatasetV2(transform=test_trsf_main, video_time_clips=train_frames_per_clip, memory_size=memory_size, type=type)
    
    return train_ds, test_ds


def get_test_dataset(img_size, test_root, frames_per_clip, type='normal'):
    test_trsf_main = Compose_imglabel([
        Resize_video(img_size, img_size),
        toTensor_video(),
        Normalize_video(mean, std)
    ])
    test_ds = TestDataset(test_root, transform=test_trsf_main, video_time_clips=frames_per_clip, type=type)
    return test_ds


def get_test_datasetV2(img_size, test_root, frames_per_clip, memory_size, type='normal'):
    test_trsf_main = Compose_imglabel([
        Resize_video(img_size, img_size),
        toTensor_video(),
        Normalize_video(mean, std)
    ])
    test_ds = TestDatasetV2(test_root, transform=test_trsf_main, video_time_clips=frames_per_clip, memory_size=memory_size, type=type)
    return test_ds


def get_sun_image(img_size, video_time_clips, sup_batch_size, unsup_batch_size, type='normal'):
    train_sup_ds = TrainDatasetV2_image(video_time_clips=video_time_clips, mode='train', type=type, sup=True, img_size=img_size)
    train_unsup_ds = TrainDatasetV2_image(video_time_clips=video_time_clips, mode='train', type=type, sup=False, img_size=img_size)
    val_ds = TrainDatasetV2_image(video_time_clips=10, mode='val', type=type, sup=True, img_size=img_size)
    
    train_sup_dl = DataLoader(train_sup_ds, batch_size=sup_batch_size, num_workers=4, shuffle=True)
    train_unsup_dl = DataLoader(train_unsup_ds, batch_size=unsup_batch_size, num_workers=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=False)
    
    return train_sup_dl, train_unsup_dl, val_dl

def convert_normalized_to_rgb(Fs):
    b, T, c, h, w = Fs.shape
    for t in range(T):
        for i in range(3):
            Fs[:, t, i] *= float(std[i])
        for i in range(3):
            Fs[:, t, i] += float(mean[i])
    return Fs * 255


if __name__ == "__main__":
    # train_ds, val_ds = get_sun_video_datasetV2(352, train_frames_per_clip=10, memory_size=3, type='small')
    # train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    # for data in train_dl:
    #     img = data['img']
    #     img = convert_normalized_to_rgb(img)
    #     break
    
    # demo = img[0, 0].permute(1, 2, 0).numpy()
    # print(demo.shape)
    # demo = Image.fromarray(demo.astype(np.uint8))
    # demo.save('/ssd/cjx/semi-supervised/cjx-semi/demo.png')
    loader = get_test_datasetV2(img_size=352, test_root="test_easy_seen", frames_per_clip=10, memory_size=3, type='normal')
    loader = DataLoader(loader, batch_size=2, shuffle=False, num_workers=8)
    for data in loader:
        img = data['img']
        path_li = data['path']
        print(img.shape)
        print(len(path_li))
        break
