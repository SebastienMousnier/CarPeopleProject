clear all;close all;clc;

im = imread('vanishing_test.png');
im = rgb2gray(im);
%imshow(im);

new = edge(im,'Canny',treshold);
imshow(new);
