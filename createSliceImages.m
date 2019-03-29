%%%Creating slice images

function createSliceImages()

    las = lasdata('000029.las');
    classes = las.get_classification();

    res = [256, 256];
    images = zeros(10000, res(1), res(2));
    imgCount = 0;
    turret = zeros(10000, 1);
    turretImages = zeros(10000, res(1), res(2));
    
    lasx = las.x - min(las.x);
    lasy = las.y - min(las.y);
    lasz = las.z - min(las.z);
    
    voxelSize = 0.2;
    sliceDepth = 5;

    for angle = linspace(0, pi/2, 4)

        m = rotz(angle);
        fprintf('Rotating %f\n', angle);

        xyz = [lasx, lasy, lasz];
        xyz = xyz * m;
        
        x = xyz(:,1);
        for d = min(x):sliceDepth:max(x)
            
            p = x >= d & x < d + sliceDepth;
            if sum(p) == 0
                continue;
            end
            
            sliceYZ = xyz(p,2:3);
            sliceClass = classes(p,:);
            turretPoints = sliceClass == 16 | sliceClass == 19;
            
            [img, imgTurret] = createImage(sliceYZ, voxelSize, turretPoints);
            tiles = tileImage(img, res);
            tilesT = tileImage(imgTurret, res);
            
            for i = 1:length(tilesT)
                if sum(sum(tiles{i})) == 0
                    continue;
                end
                
                imgCount = imgCount + 1;
                turret(imgCount) = sum( sum(tilesT{i}) ) > 10;
                images(imgCount,:,:) = tiles{i};
                turretImages(imgCount,:,:) = tilesT{i};
            end
        end
    end
    
    images = images(1:imgCount,:,:);
    turret = turret(1:imgCount,:,:);
    turretImages = turretImages(1:imgCount,:,:);
    
    s = sprintf('images_cnn_%d_%d.mat', res(1), res(2));
    save(s, 'images', 'turret', 'turretImages');

end

function [img, imgFilter] = createImage(xy, voxelSize, filter)

    x = xy(:,1);
    y = xy(:,2);

    voxelIndexX = floor((x - min(x)) / voxelSize)+1;
    voxelIndexY = floor((y - min(y)) / voxelSize)+1;

    img = zeros(max(voxelIndexX), max(voxelIndexY));
    for i = 1:length(voxelIndexX)
        img(voxelIndexX(i), voxelIndexY(i)) = img(voxelIndexX(i), voxelIndexY(i)) + 1;
    end
    img = rot90(img);
    
    imgFilter = zeros(max(voxelIndexX), max(voxelIndexY));
    for i = 1:length(voxelIndexX)
        if filter(i)
            imgFilter(voxelIndexX(i), voxelIndexY(i)) = imgFilter(voxelIndexX(i), voxelIndexY(i)) + 1;
        end
    end
    imgFilter = rot90(imgFilter);
    
    
%     plot(x,y, '.'); 
%     figure;
%     imshow(img > 0);
%     figure;
%     imshow(imgFilter > 0);
%     close all;
end

function tiles = tileImage(img, res)

    if size(img,1) < res(1) ||  size(img,2) < res(2)
        img(res(1), res(2)) = 0;
    end

    tiles = {};
    
    cutX = unique([res(1):res(1):size(img,1), size(img,1)]);
    cutY = unique([res(2):res(2):size(img,2), size(img,2)]);
    
    for i = cutX
        for j = cutY
            tiles{end+1} = img(i-res(1)+1 : i, j-res(2)+1 :j);
        end
    end
    
end
