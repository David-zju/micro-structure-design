% tic

%get all properties of unit cells by their .txt files
%the properties of the unit cells with same thickness will be saved in a
%.txt file
aThick = [1,1.1,1.2,1.3,1.4,1.6];
%aThick = 1;
for t = 1:length(aThick)
    if(~(aThick(t) == 1)) strThick = num2str(aThick(t),'%.1f'); else strThick = num2str(aThick(t)); end
    dirpath = ['C:\Users\David\projects\deeplearning\micro-structure\data_process\CaeModels\CubicPLS_T',strThick ,'\N1422\'];
    % Tclfile_path = 'E:\Projects\CAE_projects\PLS_analysis\PropSpace\CubicPLS_T0.2\'
    Tclfile_path = ['C:\Users\David\projects\deeplearning\micro-structure\data_process\CaeModels\'];

    dirModel=dir( sprintf('%s*.txt',dirpath) ); %N183L24/E70v0.3d2.7
    numModelfile=size(dirModel,1); 
    fprintf('numfile of plate unit cell mmodels is:%d\n',numModelfile);
    property = ["         Name                 E11            V12            G12            C11            C12            C44        RelDensity      AniRatio         mass      ",];
    for k = 1:numModelfile
        address=strcat(dirpath,dirModel(k).name);
        fid = fopen(address,'r');
        tline = fgetl(fid);
        while ischar(tline)
           if (contains(tline,"L"))
               property(k+1) = tline;
           end
           tline = fgetl(fid);
        end
        fclose(fid);
    end

    strfilename = ['T',strThick,'L20E70v0.3d2.7N1422_dirProp_Space.txt'] 
    fid=fopen([Tclfile_path,strfilename],'w');%写入文件路径
    for k=1:(numModelfile+1)
        fprintf(fid,'%s\r\n',property(k));
    end
    fclose(fid);
    
end


%based on all .txt files of the unit cells,  save all properties with .mat file
pls_prop = [ ["",0.0,0.0,0.0,0.0,0.0,0.0] ];
address = 'PLS_property/CubicPLS_T0.2/E70v0.3d2.7N1422L20_dirProp_Space.txt';%E70v0.3d2.7CombineN2148L24_dirProp_Space  %OrthoPLS_D0.064/PLS_DirProperty_Space.txt
fid = fopen(address,'r');
tline = fgetl(fid);
icount = 0;
dStanderE = 0;
while ischar(tline)
    if (contains(tline,"L"))  %50G0T43-2
        icount = icount + 1;
        PLSname = strtrim(tline(1:35));
%         fprintf(PLSname);fprintf(pls_prop(icount,1));
        pls_prop(icount,1) = PLSname;
        pls_prop(icount,2) = strtrim(tline(36:50));   %E
        pls_prop(icount,3) = strtrim(tline(51:65));  %v
        pls_prop(icount,4) = strtrim(tline(66:80));  %G
        pls_prop(icount,5) = strtrim(tline(126:140));%density
%         G = str2double(pls_prop(icount,4));
% %         pls_prop(icount,4) = num2str(2*G);  
%         Er = str2double(pls_prop(icount,2))/( Es*str2double(pls_prop(icount,5)));
%         Gr = G/( Gs*str2double(pls_prop(icount,5)));
%         pls_prop(icount,2) = num2str(Er);
%         pls_prop(icount,4) = num2str(Gr); 
    end
    tline = fgetl(fid);
end
fclose(fid);
save("output_mat/CubicPLS_T0.2/pls_prop_E70v0.3d2.7N1422L20.mat",'pls_prop'); %pls_prop_E70v0.3d2.7FinalGrpUcN418L24.matOrthoPLS_D0.064


% %get the namelist of all unit cells
% dirpath = 'E:\Projects\CAD_projects\PLS_generation\PlateUnitCell_generation\testdata\PLSlibrary_txt\Test\';
% dirModel=dir( sprintf('%s*.txt',dirpath) ); %N183L24/E70v0.3d2.7
% numModelfile=size(dirModel,1); 
% fprintf('the number of the namelist of plate unit cell mmodels is:%d\n',numModelfile);
% for n = 1:numModelfile
%     strName = extractBefore(strtrim( dirModel(n).name ),"_D0") + "S";
%     strUcName(n,1) = strName;
% end
% save("output_mat/CubicPLS_IntfType/"+ 'strUcName' +".mat",'strUcName');


 
 
