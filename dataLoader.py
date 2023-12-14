from imports import *

HateSpeech = "/content/drive/MyDrive/hateSpeechDetection/Hate Speech"
NoHateSpeech = "/content/drive/MyDrive/hateSpeechDetection/No Hate Speech"

def extract_text(path):
  image_path=path
  extractedInformation = pytesseract.image_to_string(Image.open(image_path))
  return extractedInformation



class CustomDataset(Dataset):
    def __init__(self, image_paths, texts, labels, transform=None, 
                 xlnet_tokenizer=None,bert_tokenizer=None,max_len=1024):
      self.image_paths = image_paths
      self.labels = labels
      self.max_len = max_len
      self.bert_tokenizer = bert_tokenizer
      self.xlnet_tokenizer = xlnet_tokenizer
      self.transform=transform

    def __len__(self):
      return len(os.listdir(self.image_paths))

    def __getitem__(self, idx):
      image = Image.open(self.image_paths[idx]).convert('RGB')
      label = self.labels[idx]
      extracted_text = extract_text(self.image_paths[idx])

      if self.transform:
        image = self.transform(image)

      bert_encoding = self.bert_tokenizer.encode_plus(
                                text = extracted_text,
                                add_special_tokens = True,
                                max_length = self.max_token_len,
                                return_token_type_ids = False,
                                padding="max_length",
                                truncation=True,
                                return_attention_mask = True,
                                return_tensors='pt',
                                # is_split_into_words=True
                        )

      xlnet_encoding = self.xlnet_tokenizer.encode_plus(
                                text = extracted_text,
                                add_special_tokens = True,
                                max_length = self.max_token_len,
                                return_token_type_ids = False,
                                padding="max_length",
                                truncation=True,
                                return_attention_mask = True,
                                return_tensors='pt',
                                # is_split_into_words=True
                        )
      return image, label, bert_encoding, xlnet_encoding
        
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
])

xlnet_tokenizer = transformers.XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

batch_size = 16
learning_rate = 0.001
num_epochs = 10
num_features = 512
num_classes=2

dataset1 = CustomDataset(HateSpeech,1,transform,xlnet_tokenizer, bert_tokenizer)
dataset2 = CustomDataset(NoHateSpeech,0,transform,xlnet_tokenizer, bert_tokenizer)

combined_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
train_ratio = 0.8
dataset_size = len(combined_dataset)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
