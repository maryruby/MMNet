f = 0.375:0.01:0.565;
upch3 = [-25 -25.9 -26.8 -27.9 -28.9 -29.8 -30.9 -31.9 -32.4 -33 -32.7 -32.4 -31.7 -31 -30.1 -29.2 -28.5 -27.8 -27.0 -26.3];
upch2 = [-8.9 -9.8 -10.9 -12.1 -13.7 -15.2 -17.0 -19.7 -25.0 -25.3 -25.0 -20.5 -17.9 -16.5 -15.1 -13.9 -12.8 -11.7 -11.0 -10.3];
disp('Двухсигнальная частотная селективность');
fprintf('%8s%8s%8s%8s%8s\n','f','upch2','upch3');
for i=1:20
    if (upch2(i)==20)||(upch3(i)==20)
        fprintf('%8d%8.2f%8.2f%8.2f%8.2f\n', f(i), upch2(i), upch3(i));
    end
end