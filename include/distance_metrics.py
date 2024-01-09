
from scipy.stats import wasserstein_distance
import seaborn as sns
from histograms import *

def EMD(input,do_print = False):
    if len(input) == 6:
        histograms = {}
        distances = {}
        histograms['image_a'] = {'channel_1': input[0], 'channel_2': input[1], 'channel_3': input[2]}
        histograms['image_b'] = {'channel_1': input[3], 'channel_2': input[4], 'channel_3': input[5]}

        bins_1 = np.arange(len(input[0]))
        bins_2 = np.arange(len(input[1]))
        bins_3 = np.arange(len(input[2]))

        distances['channel_1'] = wasserstein_distance(bins_1,bins_1,histograms['image_a']['channel_1'], histograms['image_b']['channel_1'])
        distances['channel_2'] = wasserstein_distance(bins_2,bins_2,histograms['image_a']['channel_2'], histograms['image_b']['channel_2'])
        distances['channel_3'] = wasserstein_distance(bins_3,bins_3,histograms['image_a']['channel_3'], histograms['image_b']['channel_3'])
        distances['mean'] =  np.mean([distances['channel_1'], distances['channel_2'], distances['channel_3']])

        if do_print:
            print("Wasserstein Distances:", distances)
        return distances

    elif len(input) == 2:
        hist_img_a = input[0]
        hist_img_b = input[1]

        # Wassertein Distance, we need to specify the number of bins
        bins = np.arange(len(hist_img_a))
        wa_distance = wasserstein_distance(bins,bins,hist_img_a, hist_img_b)

        if do_print:
            print("Wasserstein Distance:", wa_distance)
        return wa_distance

    else:
        raise ValueError("Wrong number of histograms inserted")

def Chi_S(input,do_print = False):
    if len(input) == 6:
        histograms = {}
        distances = {}
        histograms['image_a'] = {'channel_1': input[0], 'channel_2': input[1], 'channel_3': input[2]}
        histograms['image_b'] = {'channel_1': input[3], 'channel_2': input[4], 'channel_3': input[5]}

        bins_1 = np.arange(len(input[0]))
        bins_2 = np.arange(len(input[1]))
        bins_3 = np.arange(len(input[2]))

        distances['channel_1'] = cv2.compareHist(histograms['image_a']['channel_1'], histograms['image_b']['channel_1'], cv2.HISTCMP_CHISQR)
        distances['channel_2'] = cv2.compareHist(histograms['image_a']['channel_2'], histograms['image_b']['channel_2'], cv2.HISTCMP_CHISQR)
        distances['channel_3'] = cv2.compareHist(histograms['image_a']['channel_3'], histograms['image_b']['channel_3'], cv2.HISTCMP_CHISQR)
        distances['mean'] = np.mean([distances['channel_1'], distances['channel_2'], distances['channel_3']])

        if do_print:
            print("Chi-Squared Distances:", distances)
        return distances

    elif len(input) == 2:
        hist_img_a = input[0]
        hist_img_b = input[1]

        # Chi-Squared Distance
        chi_squared_distance = cv2.compareHist(hist_img_a, hist_img_b, cv2.HISTCMP_CHISQR)
        if do_print:
            print("Chi-Squared Distance:", chi_squared_distance)
        return chi_squared_distance

    else:
        raise ValueError("Wrong number of histograms inserted")

def KL(input,do_print = False):
    if len(input) == 6:
        histograms = {}
        distances = {}
        histograms['image_a'] = {'channel_1': input[0], 'channel_2': input[1], 'channel_3': input[2]}
        histograms['image_b'] = {'channel_1': input[3], 'channel_2': input[4], 'channel_3': input[5]}

        bins_1 = np.arange(len(input[0]))
        bins_2 = np.arange(len(input[1]))
        bins_3 = np.arange(len(input[2]))

        distances['channel_1'] = cv2.compareHist(histograms['image_a']['channel_1'], histograms['image_b']['channel_1'], cv2.HISTCMP_KL_DIV)
        distances['channel_2'] = cv2.compareHist(histograms['image_a']['channel_2'], histograms['image_b']['channel_2'], cv2.HISTCMP_KL_DIV)
        distances['channel_3'] = cv2.compareHist(histograms['image_a']['channel_3'], histograms['image_b']['channel_3'], cv2.HISTCMP_KL_DIV)
        distances['mean'] = np.mean([distances['channel_1'], distances['channel_2'], distances['channel_3']])

        if do_print:
            print("Kullback-Leibler Divergences:", distances)
        return distances

    elif len(input) == 2:
        hist_img_a = input[0]
        hist_img_b = input[1]

        # Kullback-Leibler Divergence
        kl_divergence = cv2.compareHist(hist_img_a, hist_img_b, cv2.HISTCMP_KL_DIV)
        if do_print:
            print("Kullback-Leibler Divergence:", kl_divergence)
        return kl_divergence

    else:
        raise ValueError("Wrong number of histograms inserted")
    
def plot_matrix(matrix, title, driver_numbers, figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, cmap='viridis', annot=True, fmt=".0f",
                xticklabels=driver_numbers, yticklabels=driver_numbers,
                annot_kws={"size": 10}, ax=ax)
    plt.title(title)
    plt.show()
    return


def compare_images(dict_image_paths, dimension, hist_type):
    num_images = len(dict_image_paths)
    driver_numbers = [item['driver_num'] for item in dict_image_paths]
    # Create a matrix to store the comparison results
    earth_mover_matrix = np.zeros((num_images, num_images))
    chi_squared_matrix = np.zeros((num_images, num_images))
    kl_divergence_matrix = np.zeros((num_images, num_images))

    for i in range(num_images - 1):
        for j in range(i + 1, num_images):
            # Calculate histograms for the current pair of images
            output_histograms = retrieve_2_histograms(dict_image_paths[i]['img_path'],dict_image_paths[j]['img_path'], dimension=dimension, hist_type= hist_type)
            symmetric_histograms = retrieve_2_histograms(dict_image_paths[j]['img_path'],dict_image_paths[i]['img_path'], dimension=dimension, hist_type= hist_type)

            # Compare histograms using the three distance metrics
            earth_mover_distance = EMD(output_histograms)
            chi_squared_distance = Chi_S(output_histograms)
            kl_divergence = KL(output_histograms)
            # Compute the Symetrics
            chi_s_symm = Chi_S(symmetric_histograms)
            kl_symm = KL(symmetric_histograms)
            
            if dimension == '3D' or dimension == '2D':
                earth_mover_matrix[i, j] = earth_mover_distance
                chi_squared_matrix[i, j] = chi_squared_distance
                kl_divergence_matrix[i, j] = kl_divergence
                #KL and Chi-Squared are not symmetric
                chi_squared_matrix[j, i] = chi_s_symm
                kl_divergence_matrix[j, i] = kl_symm

            elif dimension == '1D':
                earth_mover_matrix[i, j] = earth_mover_distance['mean']
                chi_squared_matrix[i, j] = chi_squared_distance['mean']
                kl_divergence_matrix[i, j] = kl_divergence['mean']
                #KL and Chi-Squared are not symmetric
                chi_squared_matrix[j, i] = chi_s_symm['mean']
                kl_divergence_matrix[j, i] = kl_symm['mean']
            else:
                raise ValueError("Wrong number of dimensions")

    # Plot the matrices using seaborn
    plot_matrix(earth_mover_matrix, "Earth Mover's Distance Matrix", driver_numbers)
    plot_matrix(chi_squared_matrix, "Chi-Squared Distance Matrix", driver_numbers)
    plot_matrix(kl_divergence_matrix, "Kullback-Leibler Divergence Matrix", driver_numbers)
    return
