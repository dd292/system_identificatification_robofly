function error = MSE(actual, calculated)
    error= mean(((actual- calculated).^2))
end