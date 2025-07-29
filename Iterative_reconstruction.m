clc;
clear all;
close all;
%%
% Load and prepare hologram
Ho = double(imread('.\data\frame48960.tif'));
Ho = Ho(651:end,1:1730);
[M,N] = size(Ho);
pixel_pitch = 5.5e-6/55;
lambda = 532*10^(-9);      % wavelength in meter
h1 = M*pixel_pitch;        % length of hologram
h2 = N*pixel_pitch;
figure, imagesc(Ho); colormap gray; title('Original Hologram');
%%
% Fourier transform of hologram
spectrum_o = FT2Dc(Ho);
figure, imagesc(log(abs(spectrum_o))); colormap jet; title('Spectrum (log)');
%%
% Select and filter the +1 order
filtered = imrect();
pos_rect = round(filtered.getPosition());
filtered_spectrum_o = spectrum_o(pos_rect(2) + (0:pos_rect(4)), pos_rect(1) + (0:pos_rect(3)));
figure, imagesc(log(abs(filtered_spectrum_o))); colormap gray; title('Filtered +1 Order (log)');
%%
% Pad the filtered spectrum to match original dimensions
p = M; q = N;
[m,n] = size(filtered_spectrum_o);
filtered_spectrum_o_upd = padarray(filtered_spectrum_o, [floor((p-m)/2) floor((q-n)/2)], 0, 'post');
filtered_spectrum_o_upd = padarray(filtered_spectrum_o_upd, [ceil((p-m)/2) ceil((q-n)/2)], 0, 'pre');
figure, imagesc(log(abs(filtered_spectrum_o_upd))); colormap gray; title('Padded Filtered Spectrum (log)');
filtered_spectrum_o = filtered_spectrum_o_upd;
%%
% Obtain hologram and phase in image plane
filtered_holo = fftshift(ifft2(fftshift(filtered_spectrum_o)));
figure, imagesc(abs(filtered_holo)); colormap gray; title('Filtered Hologram Amplitude');
phase_holo = exp(1i*angle(filtered_holo));
figure, imagesc(angle(filtered_holo)); colormap gray; title('Filtered Hologram Phase (with aberrations)');
%%
% PCA-based phase aberration compensation
tic
[N_pca, M_pca] = size(filtered_spectrum_o);
IFFT_ROI = fftshift(ifft2(fftshift(filtered_spectrum_o)));
ConjPhase = exp(1i*angle(IFFT_ROI));
[U, S, V] = svd(ConjPhase);
SS = zeros(N_pca, M_pca);
num = 1; % number of principal components to use

% Take first 'num' values
for i = 1:num
    SS(i, i) = S(i, i);
end

% Process U components
Unwrap_U = unwrap(angle(U(:, 1:2)));
SF_U1 = polyfit(1:N_pca, Unwrap_U(:, 1)', 2); % second degree
SF_U2 = polyfit(1:N_pca, Unwrap_U(:, 2)', 2); % second degree
EstimatedSF_U1 = polyval(SF_U1, 1:N_pca);
EstimatedSF_U2 = polyval(SF_U2, 1:N_pca);
New_U1 = exp(1i*(EstimatedSF_U1'));
New_U2 = exp(1i*(EstimatedSF_U2'));
U = U*0;
U(:, 1:2) = [New_U1 New_U2];

% Process V components
Unwrap_V = unwrap(angle(V(:, 1:2)));
SF_V1 = polyfit(1:M_pca, Unwrap_V(:, 1).', 2); % second degree
SF_V2 = polyfit(1:M_pca, Unwrap_V(:, 2).', 2); % second degree
EstimatedSF_V1 = polyval(SF_V1, 1:M_pca);
EstimatedSF_V2 = polyval(SF_V2, 1:M_pca);
New_V1 = exp(1i*(EstimatedSF_V1.'));
New_V2 = exp(1i*(EstimatedSF_V2.'));
V = V*0;
V(:, 1:2) = [New_V1 New_V2];

% Reconstruct the aberration phase
Z = U*SS*V';
figure, imagesc(angle(Z)); colormap gray; title('Estimated Aberration Phase');
toc
%%
% Apply the aberration correction to the filtered spectrum
Psi_pca = fftshift(fft2(fftshift(IFFT_ROI.*conj(Z))));
figure, imagesc(log(abs(Psi_pca))); colormap gray; title('Aberration-Corrected Spectrum (log)');
%%
corrected_holo = fftshift(ifft2(fftshift(Psi_pca)));
figure, imagesc(angle(corrected_holo)); colormap gray; title('Corrected Hologram Phase');
%%
% Define reconstruction distances
z1 = -27e-6;  % First beam focus distance
z2 = -5e-6;   % Second beam focus distance

% M = 2048;
% N = 2048;
% Create propagators for both distances
prop_z1 = Propagator(M, N, lambda, h1, h2, z1);
prop_z2 = Propagator(M, N, lambda, h1, h2, z2);
prop_neg_z1 = Propagator(M, N, lambda, h1, h2, -z1);
prop_neg_z2 = Propagator(M, N, lambda, h1, h2, -z2);
%% Conventional
conv_z1 = fftshift(ifft2(fftshift(Psi_pca.*prop_neg_z1)));
z1_ph = angle(conv_z1);
unwrap_z1 = unwrap_TIE(z1_ph);
conv_z2 = fftshift(ifft2(fftshift(Psi_pca.*prop_neg_z2)));
z2_ph = angle(conv_z2);
unwrap_z2 = unwrap_TIE(z2_ph);

figure, imagesc(unwrap_z1); colormap jet; colorbar;
figure, imagesc(unwrap_z2); colormap jet; colorbar;
%%
filtered_spectrum_current = Psi_pca;
max_Iterations = 4;  % Increase number of iterations for better results

for iter = 1:max_Iterations
    fprintf('Iteration: %d\n', iter);
    
    % Propagate to first focus plane (z1)
    rec_z1 = fftshift(ifft2(fftshift(filtered_spectrum_current.*prop_neg_z1)));
    amp_z1 = abs(rec_z1);
    ph_z1 = angle(rec_z1);
    ph_unwrap_z1 = unwrap_TIE(ph_z1);
    constrained_ph_z1 = ph_unwrap_z1;
    constrained_ph_z1(constrained_ph_z1 < 0) = 0;
    modified_field_z1 = amp_z1 .* exp(1i .* constrained_ph_z1);
    back_to_holo_z1 = IFT2Dc(FT2Dc(modified_field_z1) .* prop_z1);
    filtered_z1 = (exp(1i.*angle(corrected_holo))) .* abs(back_to_holo_z1).* conj(exp(1i.*angle(back_to_holo_z1)));
%     filtered_z1 = abs(corrected_holo).*(exp(1i.*angle(corrected_holo))) .* abs(back_to_holo_z1).* conj(exp(1i.*angle(back_to_holo_z1)));
    %     figure, imagesc(angle(filtered_z1)); colormap gray;
    
    rec_z2 = fftshift(ifft2(fftshift(FT2Dc(filtered_z1).*prop_neg_z2)));
    amp_z2 = abs(rec_z2);
    ph_z2 = angle(rec_z2);
%     figure, imagesc(ph_z2); colormap gray;
    ph_unwrap_z2 = unwrap_TIE(ph_z2);
%     figure, imagesc(ph_unwrap_z2); colormap jet;
    
    constrained_ph_z2 = ph_unwrap_z2;
    constrained_ph_z2(constrained_ph_z2 > 0) = 0;
    modified_field_z2 = amp_z2 .* exp(1i .* constrained_ph_z2);
    back_to_holo_z2 = IFT2Dc(FT2Dc(modified_field_z2) .* prop_z2);
    filtered_z2 = (exp(1i.*angle(corrected_holo))).* abs(conv_z2).* conj(exp(1i.*angle(back_to_holo_z2)));
    filtered_z3 = (exp(1i.*angle(corrected_holo))).* abs(back_to_holo_z2).* conj(exp(1i.*angle(back_to_holo_z2)));
%     filtered_z2 = abs(corrected_holo).*(exp(1i.*angle(corrected_holo))).* abs(back_to_holo_z2).* conj(exp(1i.*angle(back_to_holo_z2)));
%     figure, imagesc(angle(filtered_z2)); colormap gray;
    
    filtered_spectrum_current = FT2Dc(filtered_z3);
end

final_rec_z1 = fftshift(ifft2(fftshift(filtered_spectrum_current.*prop_neg_z1)));
final_amp_z1 = abs(final_rec_z1);
final_ph_z1 = angle(final_rec_z1);
final_ph_unwrap_z1 = unwrap_TIE(final_ph_z1);

% Get final field at z2 (negative phase object)
final_rec_z2 = fftshift(ifft2(fftshift(FT2Dc(filtered_z1).*prop_neg_z2)));
final_amp_z2 = abs(final_rec_z2);
final_ph_z2 = angle(final_rec_z2);
final_ph_unwrap_z2 = unwrap_TIE(final_ph_z2);

% Display final results
figure;
subplot(2,2,1);
imagesc(final_amp_z1); colormap gray; colorbar;
title('Final Amplitude at z1');

subplot(2,2,2);
imagesc(final_ph_unwrap_z1); colormap jet; colorbar;
title('Final Phase at z1 (Unwrapped)');

subplot(2,2,3);
imagesc(final_amp_z2); colormap gray; colorbar;
title('Final Amplitude at z2');

subplot(2,2,4);
imagesc(final_ph_unwrap_z2); colormap jet; colorbar;
title('Final Phase at z2 (Unwrapped)');

%% Phase plot along a line in each ROI
x1 = [502,979];
y1 = [244,244];
x2 = [1001,1478];
y2 = [704,538];
x_b1 = linspace(1,x1(2)-x1(1)+1,(x1(2)-x1(1))+1);
x_b2 = linspace(1,x2(2)-x2(1)+1,(x2(2)-x2(1))+1);
x = x_b1.*5.5/50;
x_ = x_b2.*5.5/50;

intensity_profile1 = improfile(unwrap_z1,x1, y1);
intensity_profile1_iter = improfile(final_ph_unwrap_z1,x1, y1);
figure,
plot(x, intensity_profile1, 'b', 'LineWidth', 2);
hold on;
plot(x, intensity_profile1_iter, 'r', 'LineWidth', 2);
xlabel('X(\mum)');
ylabel('Phase(radians)');
legend('Cell 1 (Conventional)', 'Cell 1 (Iterative)');


intensity_profile2 = improfile(unwrap_z2,x2, y2);
intensity_profile2_iter = improfile(final_ph_unwrap_z2,x2, y2);
figure,
plot(x_, intensity_profile2, 'b', 'LineWidth', 2);
hold on;
plot(x_, intensity_profile2_iter, 'r', 'LineWidth', 2);
xlabel('X(\mum)');
ylabel('Phase(radians)');
legend('Cell 2 (Conventional)', 'Cell 2 (Iterative)');
