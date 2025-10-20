clear all 
close all

%% Load in data and define variables (user-defined and otherwise)
%/------t-----x-----y-----z---dispx--dispy--dispz-velx---vely---velz-
path = '/home/grantblock/Research/Yellowstone/Quiver_CSV/Yellowstone_Run201_yslice_529.dat';

y_slice = true;

    %data = load('GPS_Pres11_data.dat');
    data = load(path);
    
    
    t           = data(:,1);
    x           = data(:,2);
    y           = data(:,3);
    z           = data(:,4);
    ux          = data(:,5);
    uy          = data(:,6);
    uz          = data(:,7);
    vx          = data(:,8)*3.154e+10;
    vy          = data(:,9)*3.154e+10;
    vz          = data(:,10)*3.154e+10;
    
    r_source = 6.5;

    step = 2;
    xvec = [-100:step:100];
    yvec = xvec;
    zvec = [-100:step:0];
    
    
    % xx = x(1:step:end);
    % yy = y(1:step:end);
    % zz = z(1:step:end);
    % vxx = vx(1:step:end);
    % vyy = vy(1:step:end);
    % vzz = vz(1:step:end);
    
    
    if y_slice
        [xarr, zarr] = meshgrid(yvec, zvec); % y slice
        Fvx = scatteredInterpolant(y, z, vx );
        Fvy = scatteredInterpolant(y, z, vy );
        Fvz = scatteredInterpolant(y, z, vz );
    else
        [xarr, zarr] = meshgrid(xvec, zvec); % x slice
        Fvx = scatteredInterpolant(x, z, vx );
        Fvy = scatteredInterpolant(x, z, vy );
        Fvz = scatteredInterpolant(x, z, vz );
    end

    
    vxx = Fvx(xarr, zarr);
    vyy = Fvy(xarr, zarr);
    vzz = Fvz(xarr, zarr);

     xarr = xarr/r_source;
     zarr = zarr/r_source;
    
    mask = (vzz < 0);
    mask_top = (vzz > 0);
    
    vmag = sqrt(vxx.^2 + vyy.^2 + vzz.^2);
    %vmag = sqrt(vxx.^2 + vzz.^2);
    VX = vxx./vmag;
    VY = vyy./vmag;
    VZ = vzz./vmag;

    
    %[yarr zarr] = meshgrid(yy,zz);
    
    figure();
    yyaxis left
    contourf(xarr,-zarr,vmag); hold on 
    cmap = imadjust(parula, [0, 0.6]);
    colormap(parula)
    colorbar
    caxis([0 50])
    %plot(yarr, -zarr, '.', 'markersize', [4]); hold on 
    %grid on; box on;
    hold on;


    if y_slice
        h1 = quiver(xarr, -zarr, VY,-VZ, 'white', 'filled'); 
    else
        h1 = quiver(xarr, -zarr, VX,-VZ, 'white', 'filled');
    end

    set(h1, 'LineWidth', 3.5)
    %set(h2, 'LineWidth', 3.5)
    %set(h1,'AutoScale','on', 'AutoScaleFactor', 1.0, 'LineWidth', 3.5);
    %set(h2,'AutoScale','on', 'AutoScaleFactor', 1.0, 'LineWidth', 3.5);
    set(gca, 'xlim', [-7.0, 7.0], 'ylim' ,[-0.5 3],'ydir','reverse');
    set(gca, 'Ycolor', 'black');
    set(gca, 'fontname', 'helvetica', 'fontsize', [40]);
    xlabel("r/r_{s,x}")
    ylabel("Depth/r_{s,x}")

    yyaxis right

    set(gca, 'xlim', [-7.0, 7.0], 'ylim' ,[-0.5*r_source 3*r_source],'ydir','reverse');
    set(gca, 'fontname', 'helvetica', 'fontsize', [40]);
    set(gca, 'Ycolor', 'black');
    xlabel("r/r_{s,x}")
    ylabel("Depth (km)")

    f = getframe(gca);
    %saveas(gcf, '/home/grantblock/Research/Yellowstone/Figures/run196_t=2012_xslice', 'png')

