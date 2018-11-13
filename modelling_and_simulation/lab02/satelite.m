sim('model')

R = str2num(get_param('model/Subsystem', 'R'));
Rs = str2num(get_param('model/Subsystem', 'Rs'));

for i = 1 : length(x)
    plot(x(1:i), y(1:i), 'Color','b','LineWidth',1);
    hold on
    rectangle('Position', [(0-R) (0-R), (2*R) (2*R)], ...
        'Curvature', [1 1]);
    rectangle('Position', [(x(i)-Rs) (y(i)-Rs), (2*Rs) (2*Rs)], ...
        'Curvature', [1 1]);
    axis([-10 10 -10 10]);
    daspect([1 1 1])
    hold off
    pause(2^-7);
end