import pandas as pd
from copy import deepcopy
import numpy as np
import json

import cv2
import edgeiq

class AnnotationManager:
    def __init__(self):
        self.annotations = pd.DataFrame()
        self.all_deleted = False
        self.dimensions = (0, 0)

    def get_dimensions(self):
        return self.dimensions

    def set_dimensions(self, dimensions):
        self.dimensions = dimensions

    def get_annotation_table(self):
        df = deepcopy(self.annotations)
        return df

    def update_annotation_table(self, data):
        self.annotations = pd.DataFrame()
        for item in data:
            self.annotations = self.annotations.append(pd.DataFrame({k: [v] for k, v in item.items()}))

    def update_annotations(self, shapes, label):
        if self.all_deleted:
            self.all_deleted = False
            return
        else:
            if not self.annotations.empty and len(shapes) < len(self.annotations.index):
                self.remove_annotation(shapes)
            else:
                self.add_annotation(shapes, label)

    def update_path(self, path, index):
        try:
            self.annotations.iloc[int(index)]['path'] = path
        except IndexError:
            pass

    def get_current_label(self):
        return "ROI{}".format(int(len(self.annotations) + 1))

    def add_annotation(self, shapes, label):
        if label == '' or label is None:
            label = self.get_current_label()

        new_data = {}
        new_data['label'] = label

        if shapes[-1].get('path') is not None:
            new_data['path'] = shapes[-1]['path']
        elif shapes[-1].get('x0') is not None:
            path = "{},{},{},{}".format(shapes[-1]['x0'], shapes[-1]['y0'], shapes[-1]['x1'], shapes[-1]['y1'])
            new_data['path'] = path

        self.annotations = self.annotations.append(pd.DataFrame({k: [v] for k, v in new_data.items()}))

    def remove_annotation(self, shapes):
        new_data = pd.DataFrame()
        current_shapes = self.get_all_paths(shapes)
        for _, row in self.annotations.iterrows():
            if row['path'] in current_shapes:
                new_data = new_data.append(pd.DataFrame({k: [v] for k, v in {'label': row['label'], 'path': row['path']}.items()}))
        self.annotations = new_data

    def get_all_paths(self, shapes):
        paths = []
        for shape in shapes:
            if 'path' in list(shape.keys()):
                paths.append(shape['path'])
            elif 'x0' in list(shape.keys()):
                paths.append("{},{},{},{}".format(shape['x0'], shape['y0'], shape['x1'], shape['y1']))
        return paths


    def write_image_with_boundaries(self, image, scale=None):
        if scale is None:
            scale = (1, 1)
        new_data = {}
        if not self.annotations.empty:
            paths = deepcopy(self.annotations)
            for _, row in paths.iterrows():
                new_line = row['path'].replace("M", "").replace("L",",").replace("Z","")
                line_arr = new_line.split(",")
                arr = []
                (h, w) = image.shape[:2]

                if "M" in row['path']:
                    # need to minus the y and keep the x to move from dash to cv2
                    for i in range(1, len(line_arr), 2):
                        x = int(float(line_arr[i-1])) if scale is None else int(scale[0]*float(line_arr[i-1]))
                        y = int(float(line_arr[i-1])) if scale is None else int(scale[1]*float(line_arr[i]))
                        arr.append([self.convert_x(w, x), self.convert_y(h, y)])
                    arr = self.remove_duplicates(arr)
                else:
                    (x, y, x1, y1) = [int(float(v)) for v in line_arr]
                    x = int(self.convert_x(w, x)) if scale is None else int(self.convert_x(w, x * scale[0]))
                    x1 = int(self.convert_x(w, x1)) if scale is None else int(self.convert_x(w, x1 * scale[0]))
                    y1 = int(self.convert_y(h, y1)) if scale is None else int(self.convert_y(h, y1 * scale[1]))
                    y = int(self.convert_y(h, y)) if scale is None else int(self.convert_y(h, y * scale[1]))
                    arr.append([x, y])
                    arr.append([x, y1])
                    arr.append([x1, y1])
                    arr.append([x1, y])

                if len(arr) > 0:
                    boundary = np.array([arr], np.int32)
                    boundary.reshape((-1,1,2))
                    new_data[row['label']] = boundary

                    image = cv2.polylines(image, [boundary], True,(0,0,255), thickness=3)

        return (new_data, image)

    def convert_x(self, w, value):
        if value > w:
            return w
        if value < 0:
            return 0
        return value

    def convert_y(self, h, value):
        value = h - value
        if value < 0:
            value = 0
        if value > h:
            value = h
        return value

    def remove_duplicates(self, points):
        new_points = []
        for point in points:
            if point not in new_points:
                new_points.append(point)
        return new_points

    def print_data(self, data):
        data = json.dumps(data, indent = 2)
        with open("roi.json", "w") as outfile:
            outfile.write(data)

    def remove_all_shapes(self):
        self.annotations = pd.DataFrame()


class ROIManager(AnnotationManager):
    def __init__(self):
        self.roi_sets = {}
        self.current_label = None

    def add_roi_set(self, roi_set):
        self.roi_sets[roi_set['label']] = roi_set['annotations']
        self.roi_sets[roi_set['label']].set_dimensions(self.dimensions)

    def get_annotation_table(self):
        annotation_manager = self.roi_sets.get(self.current_label)
        if annotation_manager is not None:
            table = annotation_manager.get_annotation_table()
            return table

    def remove_all_shapes(self):
        annotation_manager = self.roi_sets.get(self.current_label)
        if annotation_manager is not None:
            annotation_manager.remove_all_shapes()

    def remove_roi_set(self, label):
        del self.roi_sets[label]

    def get_all_labels(self):
        return list(self.roi_sets.keys())

    def get_annotation_manager(self, label):
        return self.roi_sets.get(label)

    def set_current_label(self, label):
        self.current_label = label

    def get_current_label(self):
        return self.current_label

    def get_next_label(self):
        annotation_manager = self.roi_sets.get(self.current_label)
        if annotation_manager is not None:
            return annotation_manager.get_current_label()

    def update_annotations(self, shapes, label):
        annotation_manager = self.roi_sets.get(self.current_label)
        if annotation_manager is not None:
            return annotation_manager.update_annotations(shapes, label)

    def update_path(self, path, index):
        annotation_manager = self.roi_sets.get(self.current_label)
        if annotation_manager is not None:
            return annotation_manager.update_path(path, index)
