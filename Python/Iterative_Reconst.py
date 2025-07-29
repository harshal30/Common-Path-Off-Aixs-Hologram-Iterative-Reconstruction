import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
from skimage.measure import profile_line
from utils import *

folder_path = r"./data/"
file_name = "frame48960.tif"
file_path = os.path.join(folder_path, file_name)
pixel_pitch = 5.5e-6 / 55
lambda_ = 532e-9
z1 = -27e-6
z2 = -5e-6
max_iterations = 4


start_time = time.time()

Ho = cv2.imread(file_path, -1).astype(np.float64)
Ho = Ho[650:, :1730]
M, N = Ho.shape
h1 = M * pixel_pitch
h2 = N * pixel_pitch

plt.figure(); plt.imshow(Ho, cmap='gray'); plt.title("Original Hologram")

spectrum_o = FT2Dc(Ho)
plt.figure(); plt.imshow(np.log(np.abs(spectrum_o)), cmap='jet'); plt.title("Spectrum (log)")
plt.show()

norm_spectrum = cv2.normalize(np.log(np.abs(spectrum_o)), None, 0, 255, cv2.NORM_MINMAX)
norm_spectrum = np.uint8(norm_spectrum)
roi = cv2.selectROI("Select +1 Order", norm_spectrum, False, False)
cv2.destroyAllWindows()
x, y, w, h = roi
filtered_spectrum_o = spectrum_o[y:y+h, x:x+w]

plt.figure(); plt.imshow(np.log(np.abs(filtered_spectrum_o)), cmap='gray'); plt.title("Filtered +1 Order (log)")


p, q = M, N
m, n = filtered_spectrum_o.shape
pad_pre = (int(np.ceil((p-m)/2)), int(np.ceil((q-n)/2)))
pad_post = (int(np.floor((p-m)/2)), int(np.floor((q-n)/2)))
filtered_spectrum_o_upd = np.pad(filtered_spectrum_o, ((pad_pre[0], pad_post[0]), (pad_pre[1], pad_post[1])), mode='constant')
filtered_spectrum_o = filtered_spectrum_o_upd

plt.figure(); plt.imshow(np.log(np.abs(filtered_spectrum_o)), cmap='gray'); plt.title("Padded Filtered Spectrum (log)")

# Filtered hologram
filtered_holo = IFT2Dc(filtered_spectrum_o)
phase_holo = np.angle(filtered_holo)
plt.figure(); plt.imshow(phase_holo, cmap='gray'); plt.title("Filtered Phase")

# PCA Phase Aberration Compensation
Psi_pca = PCA_aberration_comp(filtered_spectrum_o)

plt.figure(); plt.imshow(np.log(np.abs(Psi_pca)), cmap='gray'); plt.title("Aberration-Corrected Spectrum (log)")

corrected_holo = IFT2Dc(Psi_pca)
plt.figure(); plt.imshow(np.angle(corrected_holo), cmap='gray'); plt.title("Corrected Hologram Phase")

# Propagators
prop_z1 = Propagator(M, N, lambda_, h1, h2, z1)
prop_z2 = Propagator(M, N, lambda_, h1, h2, z2)
prop_neg_z1 = Propagator(M, N, lambda_, h1, h2, -z1)
prop_neg_z2 = Propagator(M, N, lambda_, h1, h2, -z2)

# Conventional Reconstruction
conv_z1 = IFT2Dc(Psi_pca * prop_neg_z1)
conv_z2 = IFT2Dc(Psi_pca * prop_neg_z2)
unwrap_z1 = unwrap_TIE(np.angle(conv_z1))
unwrap_z2 = unwrap_TIE(np.angle(conv_z2))

# Iterative Loop
filtered_spectrum_current = Psi_pca

for i in range(max_iterations):
    print(f"Iteration: {i+1}")
    rec_z1 = IFT2Dc(filtered_spectrum_current * prop_neg_z1)
    amp_z1 = np.abs(rec_z1)
    constrained_z1 = unwrap_TIE(np.angle(rec_z1))
    constrained_z1[constrained_z1 < 0] = 0
    mod_field_z1 = amp_z1 * np.exp(1j * constrained_z1)
    back_to_holo_z1 = IFT2Dc(FT2Dc(mod_field_z1) * prop_z1)
    filtered_z1 = np.exp(1j * np.angle(corrected_holo)) * np.abs(back_to_holo_z1) * np.conj(np.exp(1j * np.angle(back_to_holo_z1)))

    rec_z2 = IFT2Dc(FT2Dc(filtered_z1) * prop_neg_z2)
    amp_z2 = np.abs(rec_z2)
    constrained_z2 = unwrap_TIE(np.angle(rec_z2))
    constrained_z2[constrained_z2 > 0] = 0
    mod_field_z2 = amp_z2 * np.exp(1j * constrained_z2)
    back_to_holo_z2 = IFT2Dc(FT2Dc(mod_field_z2) * prop_z2)
    filtered_z3 = np.exp(1j * np.angle(corrected_holo)) * np.abs(back_to_holo_z2) * np.conj(np.exp(1j * np.angle(back_to_holo_z2)))

    filtered_spectrum_current = FT2Dc(filtered_z3)

# Final reconstructions
final_rec_z1 = IFT2Dc(filtered_spectrum_current * prop_neg_z1)
final_rec_z2 = IFT2Dc(FT2Dc(filtered_z1) * prop_neg_z2)
final_ph_unwrap_z1 = unwrap_TIE(np.angle(final_rec_z1))
final_ph_unwrap_z2 = unwrap_TIE(np.angle(final_rec_z2))

# Display final results
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1); plt.imshow(np.abs(final_rec_z1), cmap='gray'); plt.title('Final Amplitude z1'); plt.colorbar()
plt.subplot(2, 2, 2); plt.imshow(final_ph_unwrap_z1, cmap='jet'); plt.title('Final Phase z1'); plt.colorbar()
plt.subplot(2, 2, 3); plt.imshow(np.abs(final_rec_z2), cmap='gray'); plt.title('Final Amplitude z2'); plt.colorbar()
plt.subplot(2, 2, 4); plt.imshow(final_ph_unwrap_z2, cmap='jet'); plt.title('Final Phase z2'); plt.colorbar()
plt.tight_layout()

# Phase profile line plots (equivalent to improfile)
x1, y1 = [502, 979], [244, 244]
x2, y2 = [1001, 1478], [704, 538]
profile1 = profile_line(unwrap_z1, (y1[0], x1[0]), (y1[1], x1[1]))
profile1_iter = profile_line(final_ph_unwrap_z1, (y1[0], x1[0]), (y1[1], x1[1]))
profile2 = profile_line(unwrap_z2, (y2[0], x2[0]), (y2[1], x2[1]))
profile2_iter = profile_line(final_ph_unwrap_z2, (y2[0], x2[0]), (y2[1], x2[1]))

x_axis1 = np.linspace(0, len(profile1) * pixel_pitch * 1e6, len(profile1))
x_axis2 = np.linspace(0, len(profile2) * pixel_pitch * 1e6, len(profile2))

plt.figure()
plt.plot(x_axis1, profile1, 'b', label='Cell 1 (Conventional)')
plt.plot(x_axis1, profile1_iter, 'r', label='Cell 1 (Iterative)')
plt.xlabel('X (μm)')
plt.ylabel('Phase (radians)')
plt.legend()

plt.figure()
plt.plot(x_axis2, profile2, 'b', label='Cell 2 (Conventional)')
plt.plot(x_axis2, profile2_iter, 'r', label='Cell 2 (Iterative)')
plt.xlabel('X (μm)')
plt.ylabel('Phase (radians)')
plt.legend()

plt.show()

# Print total time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
