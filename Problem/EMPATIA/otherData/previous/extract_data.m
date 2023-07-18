clear all;
clc;
load('data_wemac_100Vs.mat');
table_labels = info_bbddlab_wemac.labels;
label_vect = table_labels{:,"EmocionReportada"};
id_vect = table_labels{:,"Voluntaria"};
trial_vect = table_labels{:,"Video"};
label_mat(:,1) =  id_vect;
label_mat(:,2) =  label_vect;
label_mat(:,3) =  trial_vect;

row=1;
data_volunt = [];
for v = 1:100
    for t = 1:14
        label=[];
        id = char(info_bbddlab_wemac.features{v,t}.vol_id);
        id_num = str2num(id(2:length(id)));
        for j = 1:length(id_vect)
            if label_mat(j,1)==id_num && label_mat(j,3)==t
                label = label_mat(j,2);
            end
        end
        bvp_feats = info_bbddlab_wemac.features{v,t}.EH.Video.BVP_feats; %size 31
        window = size(bvp_feats);
        window = window(1);
        gsr_feats = info_bbddlab_wemac.features{v,t}.EH.Video.GSR_feats; %size 20
        skt_feats = info_bbddlab_wemac.features{v,t}.EH.Video.SKT_feats; %size 6
        data_volunt(row:row+window-1, 1) = label;
        data_volunt(row:row+window-1, 2) = id_num;
        data_volunt(row:row+window-1, 3) = t;
        data_volunt(row:row+window-1, 4:34) = bvp_feats;
        data_volunt(row:row+window-1, 35:54) = gsr_feats;
        data_volunt(row:row+window-1, 55:60) = skt_feats;
        row = row + window;
    end
end

