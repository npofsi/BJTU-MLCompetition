import json
import os
### snli
# def read_snli_data(dir_path, data_type):
#     path = os.path.join(dir_path, f'snli_1.0_{data_type}.jsonl')
#     file = open(path, 'r', encoding='utf-8')
#     labels = []
#     text_as = []
#     text_bs = []
#     for line in file.readlines():
#         json_obj = json.loads(line)
#         label = json_obj['gold_label']
#         text_a = json_obj['sentence1']
#         text_b = json_obj['sentence2']
#         labels.append(label)
#         text_as.append(text_a)
#         text_bs.append(text_b)
#     return text_as, text_bs, labels
# def write_snli_csv(text_as, text_bs, labels, write_path):
#     write_file = open(write_path, 'w', encoding='utf-8')
#     write_file.write('\t'.join(['sentence1','sentence2','label']) + '\n')
#     for idx in range(len(labels)):
#         write_file.write('\t'.join([text_as[idx],text_bs[idx],labels[idx]]) + '\n')
        
# text_as, text_bs, labels = read_snli_data('./SNLI','test')
# write_snli_csv(text_as, text_bs, labels, f'./SNLI/test.tsv')
# text_as, text_bs, labels = read_snli_data('./SNLI','train')
# write_snli_csv(text_as, text_bs, labels, f'./SNLI/train.tsv')
# text_as, text_bs, labels = read_snli_data('./SNLI','dev')
# write_snli_csv(text_as, text_bs, labels, f'./SNLI/dev.tsv') 



def read_snli_data(dir_path, data_type):
    path = os.path.join(dir_path, f'snli_1.0_{data_type}.jsonl')
    file = open(path, 'r', encoding='utf-8')
    labels = []
    text_as = []
    text_bs = []
    for line in file.readlines():
        json_obj = json.loads(line)
        label = json_obj['gold_label']
        text_a = json_obj['sentence1']
        text_b = json_obj['sentence2']
        labels.append(label)
        text_as.append(text_a)
        text_bs.append(text_b)
    return text_as, text_bs, labels
def write_snli_csv(text_as, text_bs, labels, write_path):
    write_file = open(write_path, 'w', encoding='utf-8')
    write_file.write('\t'.join(['sentence1','sentence2','label']) + '\n')
    for idx in range(len(labels)):
        write_file.write('\t'.join([text_as[idx],text_bs[idx],labels[idx]]) + '\n')
        
text_as, text_bs, labels = read_snli_data('./SNLI','test')
write_snli_csv(text_as, text_bs, labels, f'./SNLI/test.tsv')
text_as, text_bs, labels = read_snli_data('./SNLI','train')
write_snli_csv(text_as, text_bs, labels, f'./SNLI/train.tsv')
text_as, text_bs, labels = read_snli_data('./SNLI','dev')
write_snli_csv(text_as, text_bs, labels, f'./SNLI/dev.tsv') 