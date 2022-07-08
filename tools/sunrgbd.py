import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from imageio import imread
from PIL import Image
import open3d as o3d
import scipy, glob

parser = argparse.ArgumentParser()
parser.add_argument('--path_in', type=str, default='data/scannet/ScanNet_data')
parser.add_argument('--path_out', type=str, default='data/sunrgbd')
parser.add_argument('--path_pred', type=str, default='logs/scannet/D9_2cm_eval')
parser.add_argument('--filelist', type=str, default='scannetv2_test_new.txt')
parser.add_argument('--run', type=str, default='generate_output_seg')
parser.add_argument('--label_remap', type=str, default='true')
args = parser.parse_args()

label_remap = args.label_remap.lower() == 'true'

suffix = '_vh_clean_2.ply'
subsets = {'train': 'scans', 'val': 'scans_test'}
class_labels = ('wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture')
class_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 14, 16, 24, 28, 33, 34, 36, 39)
label_dict = dict(zip(class_ids, np.arange(0, 21)))
ilabel_dict = dict(zip(np.arange(0, 21), class_ids))


def download_filelists():
  path_out = args.path_out
  zip_file = os.path.join(path_out, 'filelist.zip')
  if not os.path.exists(path_out):
    os.makedirs(path_out)

  # download
  url = 'https://www.dropbox.com/s/aeizpy34zhozrcw/scannet_filelist.zip?dl=0'
  cmd = 'wget %s -O %s' % (url, zip_file)
  print(cmd)
  os.system(cmd)

  # unzip
  cmd = 'unzip %s -d %s' % (zip_file, path_out)
  print(cmd)
  os.system(cmd)

def project(depth, K):
    Kinv = np.linalg.inv(K)
    v, u = np.meshgrid(range(depth.shape[0]),
                       range(depth.shape[1]),
                       indexing='ij')
    uv1 = np.stack([u, v, np.ones_like(u)], 0)
    xyz = np.einsum('rc,chw->rhw', Kinv, uv1 * depth)
    sobel_to_s = lambda s: np.exp(-np.abs(s) * 3)
    depth_sobel = scipy.ndimage.sobel(depth, mode='constant')
    depth_confidence = sobel_to_s(depth_sobel)
    for sigma in [3, 5, 10]:
        dmap_gauss = scipy.ndimage.gaussian_filter(depth, sigma=sigma)
        depth_sobel = scipy.ndimage.sobel(dmap_gauss, mode='constant')
        depth_confidence = np.minimum(depth_confidence,
                                      sobel_to_s(depth_sobel))
    mask = (depth_confidence.flatten() > 0.5) & (depth.flatten() > 0)

    return mask, xyz


def read_img(filename):
  file_folder = filename
  image_folder = os.path.join(file_folder, 'image')
  depth_folder = os.path.join(file_folder, 'depth')
  _, _, image_filename = next(os.walk(image_folder))
  _, _, depth_filename = next(os.walk(depth_folder))
  image = Image.open(os.path.join(image_folder, image_filename[0]))
  image = image.convert('RGB')
  image = np.asarray(image)
  depth_map = imread(os.path.join(depth_folder, depth_filename[0]))

  #rgb = image / 255. * 2.0 - 1.0
  rgb = image
  depth = depth_map / 10000.
  focal = (depth.shape[1] / 2) / np.tan(45 * np.pi / 180 / 2)
  cx = depth.shape[1] / 2
  cy = depth.shape[0] / 2
  K = np.loadtxt(os.path.join(file_folder, 'intrinsics.txt')).reshape(3, 3)
  # Project to 3D
  mask, xyz = project(depth, K)
  vertex, props = xyz.reshape(3, -1).T[mask], rgb.reshape(-1, 3)[mask]

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(vertex)
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  nv = np.array(pcd.normals)

  vertex_with_props = np.concatenate([vertex, nv, props], axis=1)
  return vertex_with_props, mask

def save_ply(point_cloud, filename):
  ncols = point_cloud.shape[1]
  py_types = (float, float, float, float, float, float,
              int, int, int, int)[:ncols]
  npy_types = [('x', 'f4'),   ('y', 'f4'),     ('z', 'f4'),
               ('nx', 'f4'),  ('ny', 'f4'),    ('nz', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
               ('label', 'u1')][:ncols]

  # format into NumPy structured array
  vertices = []
  for row_idx in range(point_cloud.shape[0]):
    point = point_cloud[row_idx]
    vertices.append(tuple(dtype(val) for dtype, val in zip(py_types, point)))
  structured_array = np.array(vertices, dtype=npy_types)
  el = PlyElement.describe(structured_array, 'vertex')

  # write ply
  PlyData([el]).write(filename)
  print('Save:', filename)


def generate_chunks(filename, point_cloud, cropsize=10.0, stride=5.0):
  vertices = point_cloud[:, :3]
  bbmin = np.min(vertices, axis=0)
  bbmax = np.max(vertices, axis=0)
  bbox = bbmax - bbmin
  inbox = bbox < cropsize
  if np.all(inbox):
    return

  chunk_id = 0
  min_size = 3000
  chunk_num = np.ceil(np.maximum(bbmax - cropsize, 0) / stride).astype(np.int32) + 1
  for i in range(chunk_num[0]):
    for j in range(chunk_num[1]):
      for k in range(chunk_num[2]):
        cmin = bbmin + np.array([i, j, k]) * stride
        cmax = cmin + cropsize
        inbox_mask = (vertices <= cmax) & (vertices >= cmin)
        inbox_mask = np.all(inbox_mask, axis=1)
        if np.sum(inbox_mask) < min_size:
          continue
        filename_out = filename.stem + '.chunk_%d.ply' % chunk_id
        save_ply(point_cloud[inbox_mask], filename.parent / filename_out)
        filename_mask = filename.stem + '.chunk_%d.mask.npy' % chunk_id
        np.save(filename.parent / filename_mask, inbox_mask)
        chunk_id += 1


def process_scannet():
  for path_out, path_in in subsets.items():
    label_lst = glob.glob(os.path.join('/media/NFS/bryan/sunrgbd-meta-data/sunrgbd_train_test_labels', '*'))
    if path_out == 'val': label_lst = label_lst[:5050]
    else : label_lst = label_lst[5050:]

    curr_path_out = Path(args.path_out) / path_out
    curr_path_out.mkdir(parents=True, exist_ok=True)

    lines = open('/media/public_dataset/SUNRGBD/' + path_out + '.txt').readlines()
    for idx, line in enumerate(lines):
      filename = os.path.join('/media/public_dataset/SUNRGBD', line.split('\n')[0])
      pointcloud, mask = read_img(filename)

      # Load label file
      label = imread(label_lst[idx]).reshape(-1)[mask]
      '''label = np.zeros((pointcloud.shape[0], 1))
      filename_label = filename.parent / (filename.stem + '.labels.ply')
      if filename_label.is_file():
        # Sanity check that the pointcloud and its label has same vertices.
        assert pointcloud.shape[0] == label_data.shape[0]
        assert np.allclose(pointcloud[:, :3], label_data[:, :3])

        import pdb
        pdb.set_trace()

        label = label_data[:, -1:]
        filename_out = curr_path_out / (filename.name[:-len(suffix)] + '.txt')
        np.savetxt(filename_out, label, fmt='%d')
        if label_remap:  # remap the files
          for i in range(label.shape[0]):
            label[i] = label_dict.get(int(label[i]), 0)'''
      assert pointcloud.shape[0] == label.shape[0]
      filename_out = curr_path_out / (os.path.splitext(label_lst[idx])[0].split('/')[-1] + '.ply')
      processed = np.concatenate((pointcloud, label[:, None]), axis=-1)
      # save the original file
      save_ply(processed, filename_out)
      # save the cropped chunks in the 10x10x10 box
      generate_chunks(filename_out, processed)


def fix_bug_files():
  bug_files = {
      'train/scene0270_00.ply': 50,
      'train/scene0270_02.ply': 50,
      'train/scene0384_00.ply': 149}
  for files, bug_index in bug_files.items():
    print(files)
    for f in Path(args.path_out).glob(files):
      pointcloud = read_ply(f)
      bug_mask = pointcloud[:, -1] == bug_index
      print(f'Fixing {f} bugged label {bug_index} x {bug_mask.sum()}')
      pointcloud[bug_mask, -1] = 0
      save_ply(pointcloud, f)


def generate_output_seg():
  # load filelist
  filename_scans = []
  with open(args.filelist, 'r') as fid:
    for line in fid:
      filename = line.split()[0]
      filename_scans.append(filename[:-4])  # remove '.ply'

  # input files
  pred_files = sorted(os.listdir(args.path_pred))
  pred_files = [f for f in pred_files if f.endswith('.npz')]
  assert len(pred_files) % len(filename_scans) == 0

  # process
  probs = {}
  for i in tqdm(range(len(pred_files)), ncols=80):
    filename_scan = filename_scans[i % len(filename_scans)]

    pred = np.load(os.path.join(args.path_pred, pred_files[i]))
    prob, inbox_mask = pred['prob'], pred['inbox_mask']
    prob0 = np.zeros([inbox_mask.shape[0], prob.shape[1]])
    prob0[inbox_mask] = prob

    if 'chunk' in filename_scan:
      filename_mask = filename_scan + '.mask.npy'
      mask = np.load(os.path.join(args.path_in, filename_mask))
      prob1 = np.zeros([mask.shape[0], prob0.shape[1]])
      prob1[mask] = prob0

      # update prob0 and filename_scan
      prob0 = prob1
      filename_scan = filename_scan[:-8]  # remove '.chunk_x'

    probs[filename_scan] = probs.get(filename_scan, 0) + prob0

  # output
  if not os.path.exists(args.path_out):
    os.makedirs(args.path_out)

  for filename, prob in tqdm(probs.items(), ncols=80):
    filename_label = filename + '.txt'
    label = np.argmax(prob, axis=1)
    for i in range(label.shape[0]):
      label[i] = ilabel_dict[label[i]]
    np.savetxt(os.path.join(args.path_out, filename_label), label, fmt='%d')


def calc_iou():
  # init
  intsc, union, accu = {}, {}, 0
  for k in class_ids[1:]:
    intsc[k] = 0
    union[k] = 0

  # load files
  pred_files = sorted(os.listdir(args.path_pred))
  pred_files = [f for f in pred_files if f.endswith('.txt')]
  for filename in tqdm(pred_files, ncols=80):
    label_pred = np.loadtxt(os.path.join(args.path_pred, filename))
    label_gt = np.loadtxt(os.path.join(args.path_in, filename))

    # omit labels out of class_ids[1:]
    mask = np.zeros_like(label_gt).astype(bool)
    for i in range(label_gt.shape[0]):
      mask[i] = label_gt[i] in class_ids[1:]
    label_pred = label_pred[mask]
    label_gt = label_gt[mask]

    ac = (label_gt == label_pred).mean()
    tqdm.write("Accu: %s, %.4f" % (filename, ac))
    accu += ac

    for k in class_ids[1:]:
      pk, lk = label_pred == k, label_gt == k
      intsc[k] += np.sum(np.logical_and(pk, lk).astype(np.float32))
      union[k] += np.sum(np.logical_or(pk,  lk).astype(np.float32))

  # iou
  iou_part = 0
  for k in class_ids[1:]:
    iou_part += intsc[k] / (union[k] + 1.0e-10)
  iou = iou_part / len(class_ids[1:])
  print('Accu: %.6f' % (accu / len(pred_files)))
  print('IoU: %.6f' % iou)


if __name__ == '__main__':
  eval('%s()' % args.run)