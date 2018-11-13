clear all

img_arr = imread('img.png');
img_arr = rgb2gray(img_arr);

COLORS_NUM = 256;
STEPS = 256;

hist = zeros(1, STEPS);
x = 0 : COLORS_NUM/STEPS : COLORS_NUM - 1;
[x_len, y_len] = size(img_arr);

for i = 1 : x_len
    for j = 1 : y_len
        val = img_arr(i, j) / (COLORS_NUM / STEPS);
        val = uint8(val);
        hist(val) = hist(val) + 1;
    end
end

subplot(2, 1, 1); imshow(img_arr); 
subplot(2, 1, 2); bar(x, hist); xlabel('oryginal');