import json
from torch import nn
import json
from pathlib import Path
import torch

class PreprocessedGLIP:
    """
        Loads COCO json predictions from each view in the current batch.
    """
    def __init__(self, pred_pth='../data/training_pred/nuscenes_glip_train_pred.pth', meta_coco='../data/training_pred/nuscenes_infos_train_mono3d.coco.json', class_names=None):

        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        
        if class_names is None:
            self.class_names = self.all_class_names

        self.glip_bbox_file = pred_pth
        self.glip_bboxes = torch.load(self.glip_bbox_file)

        self.meta_info_file = meta_coco
        with open(self.meta_info_file, 'r') as f:
            self.meta_info = json.load(f)


        print(self.meta_info['categories'])

        # self.map_catid = {(x['id'] + 1): (i + 1) for x in self.meta_info['categories'] for i, cls_name in enumerate(self.all_class_names) if cls_name == x['name']}

        # do nothing
        self.map_catid = {(i + 1): (i + 1) for i, cls_name in enumerate(self.all_class_names)}

        print('map', self.map_catid)

        self.token_to_id = {}
        self.path_to_id = {}

        for img_id, image in enumerate(self.meta_info['images']): #[img_id]['token']
            # print('image', image)
            # exit()
            self.token_to_id[image['token']] = img_id
            self.path_to_id[image['file_name']] = img_id

        # print(self.token_to_id)

    def infer_nusc(self, batch_dict):
        image_paths = batch_dict['image_paths']
        batch_size = batch_dict['batch_size']

        labels = []
        boxes = []
        scores = []
        idx = []
        cam_idx = []

        for b in range(batch_size):
            cur_paths = image_paths[b]

            for c in range(6):
                # img_id1 = (batch_dict['batch_idx'] * batch_size + b) * 6 + c
                # img_id2 = self.token_to_id[batch_dict['metadata'][b]['token']]
                img_id = self.path_to_id[str(cur_paths[c])]


                if batch_dict['metadata'][b]['token'] != self.meta_info['images'][img_id]['token']:
                    print(batch_dict['metadata'][b]['token'])

                    print(self.meta_info['images'][img_id]['token'])
                    # print('img_id1', img_id, img_id1, img_id2)

                assert batch_dict['metadata'][b]['token'] == self.meta_info['images'][img_id]['token']
                if str(batch_dict['image_paths'][b][c]) != self.meta_info['images'][img_id]['file_name']:
                    print(("Batch {} does not align with GLIP {}".format(str(batch_dict['image_paths'][b][c]), self.meta_info['images'][img_id]['file_name'])))
                    # print('img_id1', img_id, img_id1, img_id2)

                assert str(batch_dict['image_paths'][b][c]) == self.meta_info['images'][img_id]['file_name']

                c_boxes = self.glip_bboxes[img_id].bbox.reshape(-1, 4)
                c_scores = self.glip_bboxes[img_id].extra_fields['scores'].reshape(-1)
                c_labels = self.glip_bboxes[img_id].extra_fields['labels'].reshape(-1)

                for i, lbl in enumerate(c_labels):
                    # print('map')
                    c_labels[i] = self.map_catid[lbl.item()]

                # print('boxes, labels, scores', c_boxes.shape, c_labels.shape, c_scores.shape)

                boxes.append(c_boxes)
                labels.append(c_labels)
                scores.append(c_scores)
                idx.extend([b] * len(c_boxes))
                cam_idx.extend([c] * len(c_boxes))

        labels = torch.cat(labels, dim=0)
        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        idx = torch.tensor(idx)
        cam_idx = torch.tensor(cam_idx)

        return boxes, labels, scores, idx, cam_idx


    def __call__(self, batch_dict):
        if 'image_paths' in batch_dict:
            return self.infer_nusc(batch_dict)
        # elif 'frame_id' in batch_dict:
        #     return self.infer_kitti(batch_dict)
        else:
            raise TypeError('need kitti / nusc batch dict!')

class PreprocessedDetector:
    """
        Loads COCO json predictions from each view in the current batch.
    """
    def __init__(self, cam_jsons=[], class_names=[]):
        assert len(cam_jsons) > 0

        self.cam_infos = []
        self.name_to_anns = {}
        self.categories = None
        self.img_names = set()
        self.class_names = class_names
        self.infer_cam = len(cam_jsons) == 1

        # self.classes = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
        #       'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        # self.cat_to_id = {c: i for i, c in enumerate(self.classes)}

        for json_path in cam_jsons:
            with open(json_path, 'r') as f:
                d = json.load(f)

                for i in range(len(d['images'])):
                    img = d['images'][i]
                    
                    if 'name' not in img:
                        img_name = Path(img['file_name']).name
                        d['images'][i]['name'] = img_name

                self.cam_infos.append(d)

                assert self.categories is None or self.categories == d['categories'], 'categories differ!'

                if self.categories is not None:
                    assert len(self.categories) == len(d['categories'])
                self.categories = d['categories']

        cat_ids = set(x['id'] for x in self.categories)
        if self.class_names == [] or self.class_names is None:
            self.class_names = [x['name'] for x in self.categories]

        self.catid_to_classid = {x['id']: (i + 1) for x in self.categories for i, cls_name in enumerate(self.class_names) if cls_name == x['name']}
        self.wanted_catids = list(self.catid_to_classid.keys())

        if len(self.catid_to_classid) == 0:
            print('class', class_names, self.categories)
            exit()


        # print('catid_to_classid', self.catid_to_classid)
        print('categories', self.categories, self.class_names)
        # exit()

        # remap to each image
        for view_infos in self.cam_infos:
            img_id_to_name = {img['id']: img['name'] for img in view_infos['images']}

            for i, img in enumerate(view_infos['images']):
                # img_name = img_id_to_name[img['id']]

                img_name = img['name']
                self.img_names.add(img_name)

                if img_name not in self.name_to_anns:
                    self.name_to_anns[img_name] = []

            for ann in view_infos['annotations']:
                img_name = img_id_to_name[ann['image_id']]

                cat_id = ann['category_id']

                if cat_id not in cat_ids:
                    ann['category_id'] = cat_id - 1

                self.name_to_anns[img_name].append(ann)

                assert ann['category_id'] in cat_ids, f'{ann} not valid'

        # check if included extension    
        first_img_name = list(self.name_to_anns.keys())[0]
        self.incl_ext = '.jpg' in first_img_name or '.png' in first_img_name

        # print('first_img_name', first_img_name)
        # print('anns keys, total', len(self.name_to_anns), [sum([len(x) for x in self.name_to_anns.values()])])
        # print('anns', len(self.name_to_anns))

    def infer_nusc(self, batch_dict):
        image_paths = batch_dict['image_paths']
        batch_size = batch_dict['batch_size']

        labels = []
        boxes = []
        scores = []
        idx = []
        cam_idx = []

        for b in range(batch_size):
            cur_paths = image_paths[b]

            for c, path in enumerate(cur_paths):
                img_name = Path(path).stem if not self.incl_ext else Path(path).name
                # print('curr image name', img_name, 'has', len(self.name_to_anns[img_name]))
                # print('img_name', img_name, 'first key', list(self.name_to_anns.keys())[0])
                # print('in set?', img_name in self.img_names)

                if img_name not in self.name_to_anns:
                    continue

                for ann in self.name_to_anns[img_name]:
                    if ann['category_id'] not in self.wanted_catids:
                        continue
                    boxes.append(ann['bbox'])

                    labels.append(self.catid_to_classid[ann['category_id']])
                    # if using GT, will not have scores
                    score = 1.0 if 'score' not in ann else ann['score']

                    scores.append(score)
                    idx.append(b)
                    cam_idx.append(c)

                # exit()
                # batch_anns.append(self.name_to_anns[img_name])

        labels = torch.tensor(labels)
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        idx = torch.tensor(idx)
        cam_idx = torch.tensor(cam_idx)

        return boxes, labels, scores, idx, cam_idx

    def infer_kitti(self, batch_dict):
        frame_ids = batch_dict['frame_id']
        batch_size = batch_dict['batch_size']

        labels = []
        boxes = []
        scores = []
        idx = []
        cam_idx = []

        for b in range(batch_size):
            if self.incl_ext:
                cur_frame_id = frame_ids[b] + '.png'
            else:
                cur_frame_id = frame_ids[b]
        
            if cur_frame_id not in self.name_to_anns:
                raise ValueError(f'frame_id={cur_frameid} did not exist in preprocessing')

            # print('anns', self.name_to_anns[cur_frame_id])
            for ann in self.name_to_anns[cur_frame_id]:
                if ann['category_id'] not in self.wanted_catids:
                    continue
                boxes.append(ann['bbox'])
                labels.append(self.catid_to_classid[ann['category_id']])
                # if using GT, will not have scores
                score = 1.0 if 'score' not in ann else ann['score']

                scores.append(score)
                idx.append(b)
                cam_idx.append(0) # not necessary

        labels = torch.tensor(labels)
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        idx = torch.tensor(idx)
        cam_idx = torch.tensor(cam_idx)

        return boxes, labels, scores, idx, cam_idx

    def __call__(self, batch_dict):
        if 'image_paths' in batch_dict:
            return self.infer_nusc(batch_dict)
        elif 'frame_id' in batch_dict:
            return self.infer_kitti(batch_dict)
        else:
            raise TypeError('need kitti / nusc batch dict!')