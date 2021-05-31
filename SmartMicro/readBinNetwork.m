%% Matlab script to try reading in a binary file that was output from the
% python program that will be used to run the neural network
function start_monitoring_NAS()
%% Small GUI
DlgH = figure('Position',[500 500 250 100],'Name','Matlab',...
    'NumberTitle','off');
H = uicontrol('Style', 'PushButton', 'String', 'Break',...
    'Callback', 'delete(gcf)');
Text = uicontrol('Style','text','Position',[100 0 200 100],...
    'Background',[0.9 0.9 0.9],'FontSize',25);                
 drawnow               

%% While loop pulling the data from the network location
x_old = 0;
while (ishandle(H))
    fileID = fopen('//lebnas1.epfl.ch/microsc125/Watchdog/binary_output.dat','r');
    x = fread(fileID);
    fclose(fileID);
    x = size(x,1);
    Text.String = num2str(x);
    set(Text,'Background', [0.5 0.5 0.5] + (x/512));
    if x ~= x_old
        disp(x)
        disp(datetime('now','TimeZone','local','Format','HH:mm:ss.SSS'))
    end
    x_old = x;
    pause(0.01)
end

end


