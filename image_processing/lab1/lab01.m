clear all

img_array = imread('img.png');
img_array = double(img_array);

% disp('brightness parameters: ')
avg = mean(img_array, 'all');
med = median(img_array, 'all');

% disp('contrast parameters: ')
kurt = kurtosis(img_array, 1, 'all') - 3;
quart = quantile(img_array, [0.25 0.75], 'all');
quart_spac = quart(2) - quart(1);

disp('Miary jasnosci obrazu:')
disp(['Srednia: ' num2str(avg) '(im wyzsza, tym obraz jasniejszy)'])
disp(['Mediana: ' num2str(med) '(im wyzsza, tym obraz jasniejszy)'])

disp('Miary kontrastu obrazu:')
disp(['Kurtoza: ' num2str(kurt) ...
    '(jesli ujemna, to obraz bardziej skupiony niz rozklad normalny'])
disp(['Rozstep cwiartkowy: ' num2str(quart_spac) ...
    ' (im wyzszy, tym obraz ma wiekszy kontrast)'])

img_array2 = img_array + 40;
img_array3 = img_array - 40;
img_array4 = img_array * 3;
img_array5 = img_array * (1/2);
img_array6 = img_array .^ 2;
img_array7 = img_array .^ (1/2);
img_array8 = log(img_array);

img_array = uint8(img_array);
img_array2 = uint8(img_array2);
img_array3 = uint8(img_array3);
img_array4 = uint8(img_array4);
img_array5 = uint8(img_array5);
img_array6 = uint8(img_array6);
img_array7 = uint8(img_array7);
img_array8 = uint8(img_array8);

subplot(2, 4, 1); image(img_array); 
axis equal; xlabel('oryginal')
subplot(2, 4, 2); image(img_array2); 
axis equal; xlabel('dodwanie stalej')
subplot(2, 4, 3); image(img_array3); 
axis equal; xlabel('odejmowanie stalej')
subplot(2, 4, 4); image(img_array4); 
axis equal; xlabel('mnozenie > 1')
subplot(2, 4, 5); image(img_array5); 
axis equal; xlabel('mnozenie < 1')
subplot(2, 4, 6); image(img_array6); 
axis equal; xlabel('potegowanie > 1')
subplot(2, 4, 7); image(img_array7); 
axis equal; xlabel('potegowanie < 1 (sqrt)')
subplot(2, 4, 8); image(img_array8); 
axis equal; xlabel('logarytm')