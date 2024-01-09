from utility import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_crop_image(image_path):
    image = cv2.imread(image_path)
    x, y, w, h = 25, 100, 700, 700
    image_cropped = image[y:y+h, x:x+w]
    image_copy = image_cropped.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    return image_copy

def RGB_histograms(image_path, choose_dim, display=False, normalize=True):
    if choose_dim not in ["1D", "2D", "3D"]:
        raise ValueError("Expected dimension must be '1D' or '2D' or '3D")

    image_rgb = load_crop_image(image_path)

    if choose_dim == '1D':
        channels = cv2.split(image_rgb)
        hist_red = cv2.calcHist([channels[0]], [0], None, [16], [0, 256])
        hist_green = cv2.calcHist([channels[1]], [0], None, [16], [0, 256])
        hist_blue = cv2.calcHist([channels[2]], [0], None, [16], [0, 256])

        if normalize:
            hist_red = cv2.normalize(hist_red, None, alpha=1, norm_type=cv2.NORM_L1)
            hist_green = cv2.normalize(hist_green, None, alpha=1, norm_type=cv2.NORM_L1)
            hist_blue = cv2.normalize(hist_blue, None, alpha=1, norm_type=cv2.NORM_L1)

    if choose_dim == '3D':
        hist_3D = cv2.calcHist([image_rgb], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])

        if normalize:
            hist_3D = cv2.normalize(hist_3D, None, alpha=1, norm_type=cv2.NORM_L1)

        hist_3D = hist_3D.flatten()

    if display:
        plt.figure(figsize=(15, 3))
        plt.imshow(image_rgb)
        plt.axis('off')

        if choose_dim == '3D':
            plt.figure(figsize=(10, 5))
            plt.plot(hist_3D)
            plt.title('Flattened 3D RGB Histogram')
            plt.xlabel('Bin')
            plt.ylabel('Normalized Frequency')

        if choose_dim == '1D':
            plt.figure(figsize=(15, 8))
            plt.subplot(231), plt.plot(hist_red, color='red'), plt.title('Red Channel Histogram')
            plt.subplot(232), plt.plot(hist_green, color='green'), plt.title('Green Channel Histogram')
            plt.subplot(233), plt.plot(hist_blue, color='blue'), plt.title('Blue Channel Histogram')

    if choose_dim == '3D':
        return hist_3D
    if choose_dim == '1D':
        return hist_red, hist_green, hist_blue

def HSV_histogram(image_path, choose_dim, normalize=True, display=False):
    image = load_crop_image(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if choose_dim not in ["1D", "2D", "3D"]:
        raise ValueError("Expected dimension must be '1D' or '2D' or '3D")

    h_bins = 64
    s_bins = 64
    v_bins = 64
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    v_ranges = [0, 256]

    if choose_dim == '3D':
        channels = [0, 1, 2]
        histSize = [4, 4, 4]
        ranges = h_ranges + s_ranges + v_ranges
        hist_3d = cv2.calcHist([hsv_image], channels, None, histSize, ranges, accumulate=False)

        if normalize:
            hist_3d = cv2.normalize(hist_3d, None, alpha=1, norm_type=cv2.NORM_L1)

        hist_3d = hist_3d.flatten()

    if choose_dim == '2D':
        histSize = [8, 16]
        ranges = h_ranges + s_ranges
        channels = [0, 1]
        hist_2d = cv2.calcHist([hsv_image], channels, None, histSize, ranges, accumulate=False)

        if normalize:
            hist_2d = cv2.normalize(hist_2d, None, alpha=1, norm_type=cv2.NORM_L1)

    if choose_dim == '1D':
        h_channel, s_channel, v_channel = cv2.split(hsv_image)
        h_hist = cv2.calcHist([h_channel], [0], None, [h_bins], h_ranges)
        s_hist = cv2.calcHist([s_channel], [0], None, [s_bins], s_ranges)
        v_hist = cv2.calcHist([v_channel], [0], None, [v_bins], v_ranges)

        if normalize:
            h_hist = cv2.normalize(h_hist, None, alpha=1, norm_type=cv2.NORM_L1)
            s_hist = cv2.normalize(s_hist, None, alpha=1, norm_type=cv2.NORM_L1)
            v_hist = cv2.normalize(v_hist, None, alpha=1, norm_type=cv2.NORM_L1)

    if display:
        plt.figure(figsize=(15, 3))
        plt.subplot(121)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(hsv_image)
        plt.title('HSV Image')
        plt.axis('off')

        if choose_dim == '2D':
            plt.figure(figsize=(10, 5))
            plt.imshow(hist_2d, interpolation='nearest', cmap='viridis')
            plt.title('2D Histogram')
            plt.xlabel('Hue')
            plt.ylabel('Saturation')

        if choose_dim == '1D':
            plt.figure(figsize=(15, 8))
            plt.subplot(231)
            plt.plot(h_hist, color='blue')
            plt.title('Hue Histogram')
            plt.subplot(232)
            plt.plot(s_hist, color='grey')
            plt.title('Saturation Histogram')
            plt.subplot(233)
            plt.plot(v_hist, color='black')
            plt.title('Value Histogram')

        if choose_dim == '3D':
            plt.figure(figsize=(10, 5))
            plt.plot(hist_3d)
            plt.title('Flattened 3D HSV Histogram')
            plt.xlabel('Bin')
            plt.ylabel('Normalized Frequency')

    if choose_dim == '2D':
        return hist_2d.flatten()
    if choose_dim == '3D':
        return hist_3d
    if choose_dim == '1D':
        return h_hist, s_hist, v_hist

def encode_hex(color):
    b, g, r = color
    hex_value = '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)
    return hex_value

def color_distribution(image_path, number_of_colors=16, display=True):
    img = load_crop_image(image_path) / 255
    h, w, c = img.shape
    img2 = img.reshape(h * w, c)
    kmeans_cluster = KMeans(n_clusters=number_of_colors, n_init=10)
    kmeans_cluster.fit(img2)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    img3 = cluster_centers[cluster_labels].reshape(h, w, c) * 255.0
    img3 = img3.astype('uint8')
    img4 = img3.reshape(-1, 3)
    colors, counts = np.unique(img4, return_counts=True, axis=0)
    unique = list(zip(colors, counts))

    if display:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        for i, (color, count) in enumerate(unique):
            ax2.bar(i, count, color=encode_hex(color))
        ax2.set_title('Color Distribution')
        ax1.imshow(img3)
        ax1.set_title('Image with Clustered Colors')

    return unique

def retrieve_2_histograms(image_path1, image_path2, dimension, hist_type):
    if dimension not in ["1D", "2D", "3D"]:
        raise ValueError("Expected dimension must be '1D' or '2D' or '3D")

    if hist_type == 'RGB':
        if dimension == '1D':
            r1, g1, b1 = RGB_histograms(image_path1, choose_dim=dimension)
            r2, g2, b2 = RGB_histograms(image_path2, choose_dim=dimension)
            return [r1, g1, b1, r2, g2, b2]
        elif dimension == '3D':
            rgb_3d_1 = RGB_histograms(image_path1, choose_dim=dimension)
            rgb_3d_2 = RGB_histograms(image_path2, choose_dim=dimension)
            return [rgb_3d_1, rgb_3d_2]
    elif hist_type == 'HSV':
        if dimension == '1D':
            h1, s1, v1 = HSV_histogram(image_path1, choose_dim=dimension)
            h2, s2, v2 = HSV_histogram(image_path2, choose_dim=dimension)
            return [h1, s1, v1, h2, s2, v2]
        elif dimension == '2D':
            hs_1 = HSV_histogram(image_path1, choose_dim=dimension)
            hs_2 = HSV_histogram(image_path2, choose_dim=dimension)
            return [hs_1, hs_2]
        elif dimension == '3D':
            hsv_3d_1 = HSV_histogram(image_path1, choose_dim=dimension)
            hsv_3d_2 = HSV_histogram(image_path2, choose_dim=dimension)
            return [hsv_3d_1, hsv_3d_2]
    else:
        raise ValueError("Invalid histogram type")