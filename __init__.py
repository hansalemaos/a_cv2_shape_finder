import cv2
import re
from a_cv_imwrite_imread_plus import open_image_in_cv
from flatten_everything import flatten_everything
from more_itertools import chunked
import pandas as pd
import numpy as np


def get_shapes_using_ADAPTIVE_THRESH_GAUSSIAN_C(
    im,
    method=cv2.CHAIN_APPROX_SIMPLE,
    approxPolyDPvar=0.01,
    constant_subtracted=2,
    block_size=11,
    return_bw_pic=True,
):
    df, treshpic = get_shape_information_from_picture(
        im,
        method=method,
        approxPolyDPvar=approxPolyDPvar,
        method_bw=0,
        method0_constant_subtracted=constant_subtracted,
        method0_block_size=block_size,
    )
    if return_bw_pic:
        return df, treshpic
    return df, None


def get_shapes_using_ADAPTIVE_THRESH_MEAN_C(
    im,
    method=cv2.CHAIN_APPROX_SIMPLE,
    approxPolyDPvar=0.01,
    constant_subtracted=2,
    block_size=11,
    return_bw_pic=True,
):
    df, treshpic = get_shape_information_from_picture(
        im,
        method=method,
        approxPolyDPvar=approxPolyDPvar,
        method_bw=1,
        method0_constant_subtracted=constant_subtracted,
        method0_block_size=block_size,
    )
    if return_bw_pic:
        return df, treshpic
    return df, None


def get_shapes_using_THRESH_OTSU(
    im,
    method=cv2.CHAIN_APPROX_SIMPLE,
    approxPolyDPvar=0.01,
    kernel=(5, 5),
    start_thresh=127,
    end_thresh=255,
    return_bw_pic=True,
):
    df, treshpic = get_shape_information_from_picture(
        im,
        method=method,
        approxPolyDPvar=approxPolyDPvar,
        method_bw=2,
        method1_kernel=kernel,
        method1_startthresh=start_thresh,
        method1_endthresh=end_thresh,
    )
    if return_bw_pic:
        return df, treshpic
    return df, None


def get_shape_information_from_picture(
    im,
    method=cv2.CHAIN_APPROX_SIMPLE,
    approxPolyDPvar=0.01,
    method_bw=0,
    method0_constant_subtracted=2,
    method0_block_size=11,
    method1_kernel=(5, 5),
    method1_startthresh=127,
    method1_endthresh=255,
):
    imgray2 = (open_image_in_cv(im)).copy()
    grayImage_konvertiert = open_image_in_cv(imgray2, channels_in_output=2)
    if method_bw == 0:
        threshg = cv2.adaptiveThreshold(
            grayImage_konvertiert,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            method0_block_size,
            method0_constant_subtracted,
        )
    elif method_bw == 1:
        threshg = cv2.adaptiveThreshold(
            grayImage_konvertiert,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            method0_block_size,
            method0_constant_subtracted,
        )
    else:
        blur = cv2.GaussianBlur(grayImage_konvertiert.copy(), method1_kernel, 0)
        _, threshg = cv2.threshold(
            blur,
            method1_startthresh,
            method1_endthresh,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    contours, hierachy = cv2.findContours(threshg, cv2.RETR_TREE, method)
    # https://stackoverflow.com/a/59940065/15096247
    def detect_shape(approx, boundingrect):
        shape = ""
        if len(approx) == 3:
            shape = "triangle"

        # Square or rectangle
        elif len(approx) == 4:
            (x, y, w, h) = boundingrect
            ar = w / float(h)

            # A square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"

        # Pentagon
        elif len(approx) == 5:
            shape = "pentagon"

        elif len(approx) == 6:
            shape = "hexagon"

        elif len(approx) == 7:
            shape = "heptagon"
        # elif len(approx) == 8:
        #     shape = "octagon"
        # Otherwise assume as circle or oval
        else:
            (x, y, w, h) = boundingrect
            ar = w / float(h)
            shape = "circle" if 0.95 <= ar <= 1.05 else "oval"

        return shape

    coordsall = [
        (
            pd.DataFrame(
                tuple([tuple(y) for y in chunked(flatten_everything(x), 2)])
            ).assign(cv2stuff=[[p] for p in x], **{f"h{k}": v for k, v in enumerate(h)})
        )
        for x, h in zip(contours, hierachy[0])
        if len(x) >= 3
    ]

    def get_rotated_rectangle(cnt):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    def get_enclosing_circle(cnt):
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        return center, radius

    alldata = []
    rows, cols = imgray2.shape[:2]
    hcols = [
        _
        for _ in coordsall[0].columns.to_list()
        if re.search(r"^h\d+$", str(_)) is not None
    ]
    for dafr in coordsall:
        try:
            cnts = np.array([x[0] for x in dafr.cv2stuff])
            try:
                arclength = cv2.arcLength(cnts, True)
            except Exception:
                arclength = pd.NA
            cas = cv2.approxPolyDP(cnts, approxPolyDPvar * arclength, True)
            try:
                k = cv2.isContourConvex(cas)
            except Exception:
                k = pd.NA
            x = pd.NA
            y = pd.NA
            try:
                M = cv2.moments(cnts)
                if M["m00"] != 0.0:
                    x = int(M["m10"] / M["m00"])
                    y = int(M["m01"] / M["m00"])
            except Exception:
                x = pd.NA
                y = pd.NA
            try:
                area = cv2.contourArea(cas)
            except Exception:
                area = pd.NA
            try:

                boundingrect = cv2.boundingRect(cas)
            except Exception:
                boundingrect = pd.NA
            try:
                hull = cv2.convexHull(cas)
            except Exception:
                hull = pd.NA
            try:
                shap = detect_shape(approx=cas, boundingrect=boundingrect)
            except Exception:
                shap = pd.NA
            try:
                rotarect = get_rotated_rectangle(cas)
            except Exception:
                rotarect = pd.NA
            try:
                center, radius = get_enclosing_circle(cas)
            except Exception:
                center, radius = pd.NA, pd.NA
            try:
                ellipse = cv2.fitEllipse(cas)
            except Exception:
                ellipse = pd.NA
            try:
                [vx, vy, x, y] = cv2.fitLine(cas, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                coordsforline = (cols - 1, righty), (0, lefty)
            except Exception:
                coordsforline = pd.NA

            hira = tuple([dafr[mu].iloc[0] for mu in hcols])
            apptup = (
                arclength,
                k,
                x,
                y,
                area,
                boundingrect,
                hull,
                len(hull),
                len(cas),
                shap,
                rotarect,
                center,
                radius,
                ellipse,
                coordsforline,
            ) + hira
            alldata.append(apptup)
        except Exception as fe:
            continue
    df2 = pd.DataFrame(alldata)
    df2.columns = [
        "arcLength",
        "isContourConvex",
        "center_x",
        "center_y",
        "area",
        "boundingRect",
        "convexHull",
        "len_convexHull",
        "len_cnts",
        "shape",
        "rotated_rectangle",
        "minEnclosingCircle_center",
        "minEnclosingCircle_radius",
        "fitEllipse",
        "fitLine",
    ] + hcols
    df2.center_x = df2.center_x.apply(lambda x: int(x[0])).astype(np.uint32)
    df2.center_y = df2.center_y.apply(lambda x: int(x[0])).astype(np.uint32)
    boundingre = df2.boundingRect.apply(
        lambda x: pd.Series([x[0], x[1], x[0] + x[2], x[1] + x[3], x[2], x[3]])
    ).rename(
        columns={
            0: "bound_start_x",
            1: "bound_start_y",
            2: "bound_end_x",
            3: "bound_end_y",
            4: "bound_width",
            5: "bound_height",
        }
    )
    df2 = pd.concat(
        [df2[[l for l in df2.columns if l != "boundingRect"]], boundingre.copy()],
        axis=1,
    )

    df2.columns = [f"aa_{x}" for x in df2.columns]
    return df2, threshg


# if __name__ == '__main__':
#     from PrettyColorPrinter import add_printer
#
#     add_printer(True)
#     imgtocheck = r"http://clipart-library.com/img/2000719.png"
#     image2 = open_image_in_cv(imgtocheck)
#     image = image2.copy()
#     # df, bw_image = get_shape_information_from_picture(
#     #     im=image.copy(), method=cv2.CHAIN_APPROX_SIMPLE, approxPolyDPvar=0.04,
#     # )
#
#     #from a_cv2_shape_finder import get_shapes_using_ADAPTIVE_THRESH_GAUSSIAN_C
#     df,bw_pic=get_shapes_using_ADAPTIVE_THRESH_GAUSSIAN_C(
#         im=r"http://clipart-library.com/img/2000719.png",
#         method=cv2.CHAIN_APPROX_SIMPLE,
#         approxPolyDPvar=0.02,
#         constant_subtracted=2,
#         block_size=11,
#         return_bw_pic=True,
#     )
#
#     df,bw_pic=get_shapes_using_ADAPTIVE_THRESH_MEAN_C(
#         im=image.copy(),
#         method=cv2.CHAIN_APPROX_SIMPLE,
#         approxPolyDPvar=0.04,
#         constant_subtracted=2,
#         block_size=11,
#         return_bw_pic=True,
#     )
#
#
#     df,bw_pic=get_shapes_using_THRESH_OTSU(
#         im=image.copy(),
#         method=cv2.CHAIN_APPROX_SIMPLE,
#         approxPolyDPvar=0.01,
#         kernel=(1, 1),
#         start_thresh=50,
#         end_thresh=255,
#         return_bw_pic=True,
#     )
#
#     image = image2.copy()
#     for name, group in df.groupby("aa_h3"):
#
#         # if len(group) > 1:
#         #     print(group)
#         if name == 0:
#             continue
#         fabb = (
#             np.random.randint(50, 250),
#             np.random.randint(50, 250),
#             np.random.randint(50, 250),
#         )
#         for key, item in group.loc[(group.aa_area > 200) & (group.aa_shape.isin(['rectangle', 'triangle','circle','pentagon','hexagon'])) ].iterrows():
#
#             image = cv2.drawContours(
#                 image, item.aa_convexHull, -1, color=fabb, thickness=5, lineType=cv2.LINE_AA
#             )
#             image = cv2.rectangle(
#                 image,
#                 (item.aa_bound_start_x, item.aa_bound_start_y),
#                 (item.aa_bound_end_x, item.aa_bound_end_y),
#                 (0, 0, 0),
#                 3,
#             )
#             image = cv2.rectangle(
#                 image,
#                 (item.aa_bound_start_x, item.aa_bound_start_y),
#                 (item.aa_bound_end_x, item.aa_bound_end_y),
#                 fabb,
#                 2,
#             )
#             image = cv2.putText(
#                 image,
#                 f'{str(item.aa_shape)} - {name}',
#                 (item.aa_bound_start_x, item.aa_bound_start_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.4,
#                 (0, 0, 0),
#                 2,
#                 cv2.LINE_AA,
#             )
#             image = cv2.putText(
#                 image,
#                 f'{str(item.aa_shape)} - {name}',
#                 (item.aa_bound_start_x, item.aa_bound_start_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.4,
#                 fabb,
#                 1,
#                 cv2.LINE_AA,
#             )
#
#     cv2.imshow_thread([image,bw_pic])
