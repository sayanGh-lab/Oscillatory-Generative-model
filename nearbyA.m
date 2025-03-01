function [ A ] = nearbyA(n, eps_w, distance )
%NEARBY-A Summary of this function goes here
%   Detailed explanation goes here
    x = 1:n;
    xm = repmat(x, n, 1);
    A = eps_w*boolean(distance > abs(xm - xm'));
end

