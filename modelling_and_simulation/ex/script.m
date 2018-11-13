sim('model')

for i = 1 : length(x)
    plot(x(1:i), y(1:i), 'Color', 'b', 'LineWidth', 1)
    hold on
    rectangle('Position', [(x(i)-0.25) (y(i)-0.25), 0.5 0.5], ...
    'Curvature', [1 1], 'FaceColor', [1 0 0], 'EdgeColor', 'b');
    axis([0 25 0 10])
    daspect([1 1 1])
    hold off
    pause(0.005)
end