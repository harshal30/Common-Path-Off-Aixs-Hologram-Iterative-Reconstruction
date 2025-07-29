function[ p ] = Propagator(M,N,lambda,area1,area2,z)

p = zeros(M,N);

for ii = 1:M
    for jj = 1:N
        alpha = lambda*( ii - M/2-1)/area1;
        beta = lambda*( jj - N/2-1)/area2;
        if((alpha^2 + beta^2)<=1)
        p( ii , jj ) = exp(-2*pi*1i*z*sqrt(1 - alpha^2 - beta^2)/lambda);
        end  % if
    end
end 