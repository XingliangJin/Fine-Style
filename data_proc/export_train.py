import os
import sys
import numpy as np
import yaml
import argparse
import shutil
from copy import deepcopy
from os.path import join as pjoin
BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
#sys.path.insert(0, pjoin(BASEPATH, '..', '..'))

from utils.animation_data import AnimationData
from utils.load_skeleton import Skel


def pad_to_window(slice, window):
    def get_reflection(src, tlen):  # return [src-reversed][src][src-r]...
        x = src.copy()
        x = np.flip(x, axis=0)
        ret = x.copy()
        while len(ret) < tlen:
            x = np.flip(x, axis=0)
            ret = np.concatenate((ret, x), axis=0)
        ret = ret[:tlen]
        return ret

    if len(slice) >= window:
        return slice
    left_len = (window - len(slice)) // 2 + (window - len(slice)) % 2
    right_len = (window - len(slice)) // 2
    left = np.flip(get_reflection(np.flip(slice, axis=0), left_len), axis=0)
    right = get_reflection(slice, right_len)
    slice = np.concatenate([left, slice, right], axis=0)
    assert len(slice) == window
    return slice


def bvh_to_motion_and_phase(filename, downsample, skel):
    anim = AnimationData.from_BVH(filename, downsample=downsample, skel=skel)
    full = anim.get_full()  # [T, xxx]
    phases = anim.get_phases()  # [T, 1]
    return np.concatenate((full, phases), axis=-1)


def divide_clip_xia(input, window, window_step, divide):
    if not divide:  # return the whole clip
        t = ((input.shape[0]) // 4) * 4 + 4
        t = max(t, 12)
        if len(input) < t:
            input = pad_to_window(input, t)
        return [input]

    windows = []
    j = -(window // 4)
    total = len(input)
    while True:
        slice = input[max(j, 0): j + window].copy()  # remember to COPY!!
        if len(slice) < window:
            slice = pad_to_window(slice, window)
        windows.append(slice)
        j += window_step
        if total - j < (3 * window) // 4:
            break
    return windows


def divide_clip_bfa(input, window, window_step, divide):
    if not divide:  # return the whole clip
        t = ((input.shape[0]) // 4) * 4 + 4
        t = max(t, 12)
        if input.shape[0] < t:
            input = pad_to_window(input, t)
        return [input]
    windows = []
    for j in range(0, len(input) - window + 1, window_step):
        slice = input[j: j + window].copy()  # remember to COPY!!
        if len(slice) < window:
            slice = pad_to_window(slice, window)
        windows.append(slice)
    return windows


def process_file(filename, divider, window, window_step, downsample=4, skel=None, divide=True):
    input = bvh_to_motion_and_phase(filename, downsample=downsample, skel=skel)  # [T, xxx]
    return divider(input, window=window, window_step=window_step, divide=divide)


def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']


def set_init(dic, key, value):
    try:
        dic[key]
    except KeyError:
        dic[key] = value


def motion_and_phase_to_dict(fulls, style, meta, content):
    """
    fulls: a list of [T, xxx + 1] - motion and phase
    style: a *number*
    meta: a dict, e.g. {"style": "angry", "content": "walk"}
    """
    output = []
    for full in fulls:
        motion, phase = full[:, :-1], full[:, -1]
        phase_label = phase[len(phase) // 2]
        meta_copy = deepcopy(meta)
        meta_copy["phase"] = phase_label
        output.append({
            "motion": motion,
            "style": style,
            "meta": meta_copy,

            "content": content
        })
    return output


def generate_database_xia(bvh_path, output_path, window, window_step, dataset_config='xia_dataset.yml'):
    with open(dataset_config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    content_namedict = [full_name.split('_')[0] for full_name in cfg["content_full_names"]]
    content_test_cnt = cfg["content_test_cnt"]
    content_names = cfg["content_names"]
    style_names = cfg["style_names"]
#
    style_name_to_idx = {name: i for i, name in enumerate(style_names)}
    content_name_to_idx = {name: i for i, name in enumerate(content_names)}


    skel = Skel()

    bvh_files = get_bvh_files(bvh_path)

    train_inputs = []
    test_inputs = []
    trainfull_inputs = []
    test_files = []
    test_cnt = {}  # indexed by content_style

    for i, item in enumerate(bvh_files):
        print('Processing %i of %i (%s)' % (i, len(bvh_files), item))
        filename = item.split('/')[-1]
        style, content_idx, _ = filename.split('_')
        content = content_namedict[int(content_idx) - 1]
        content_style = "%s_%s" % (content, style)
#change here to add content_idx
        uclip = motion_and_phase_to_dict(process_file(item, divider=divide_clip_xia, window=window, window_step=window_step,
                                                      skel=skel, divide=False),
                                         style_name_to_idx[style],
                                         {"style": style, "content": content},content_name_to_idx[content])
        # Whether this should be a test clip
        set_init(test_cnt, content_style, 0)
        if test_cnt[content_style] < content_test_cnt[content]:
            test_cnt[content_style] += 1
            test_inputs += uclip
            test_files.append(filename)
        else:
            trainfull_inputs += uclip
            clips = motion_and_phase_to_dict(process_file(item, divider=divide_clip_xia, window=window, window_step=window_step,
                                                          skel=skel, divide=True),
                                             style_name_to_idx[style],
                                             {"style": style, "content": content},content_name_to_idx[content])
            train_inputs += clips

    data_dict = {}
    data_info = {}
    for subset, inputs in zip(["train", "test", "trainfull"], [train_inputs, test_inputs, trainfull_inputs]):
        motions = [input["motion"] for input in inputs]
        styles = [input["style"] for input in inputs]
        meta = {key: [input["meta"][key] for input in inputs] for key in inputs[0]["meta"].keys()}
# add content
        contents = [input["content"] for input in inputs]
        data_dict[subset] = {"motion": motions, "style": styles, "meta": meta, "content": contents}

        """compute meta info"""
        num_clips = len(motions)
        info = {"num_clips": num_clips,
                "distribution":
                    {style:
                         {content: len([i for i in range(num_clips) if meta["style"][i] == style and meta["content"][i] == content])
                          for content in content_names}
                     for style in style_names}
                }
        data_info[subset] = info

    np.savez_compressed(output_path + ".npz", **data_dict)

    info_file = output_path + ".info"
    data_info["test_files"] = test_files
    with open(info_file, "w") as f:
        yaml.dump(data_info, f, sort_keys=False)

    test_folder = output_path + "_test"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    for file in test_files:
        shutil.copy(pjoin(bvh_path, file), pjoin(test_folder, file))



def parse_args():
    parser = argparse.ArgumentParser("export_train")
    parser.add_argument("--dataset", type=str, default="xia")
    parser.add_argument("--bvh_path", type=str, default="styletransfer")
    parser.add_argument("--output_path", type=str, default="xia_data")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--window_step", type=int, default=8)
    parser.add_argument("--dataset_config", type=str, default='../global_info/xia_dataset.yml')
    return parser.parse_args()


def main(args):
    if args.dataset == "xia":
        generate_database_xia(bvh_path=args.bvh_path, output_path=args.output_path,
                              window=args.window, window_step=args.window_step,
                              dataset_config=args.dataset_config)
    else:
        assert 0, f'Unsupported dataset type {args.dataset}'


if __name__ == '__main__':
    args = parse_args()
    main(args)

