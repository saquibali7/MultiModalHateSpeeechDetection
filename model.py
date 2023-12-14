from imports import *

num_features = 512
num_classes = 2

class InceptionNetModel(nn.Module):
    def __init__(self, num_classes):
        super(InceptionNetModel, self).__init__()
        self.inceptionnet_model = models.inception_v3(pretrained=True)
        self.inceptionnet_model.fc = nn.Linear(self.inceptionnet_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.inceptionnet_model(x)
    
class EnsembleModel(nn.Module):
    def __init__(self, image_model, text_model_xlnet, text_model_bert, num_classes):
        super(EnsembleModel, self).__init__()
        self.image_model = image_model
        self.text_model_xlnet = text_model_xlnet
        self.text_model_bert = text_model_bert
        self.fc = nn.Linear(
            image_model.inceptionnet_model.fc.out_features +
            text_model_xlnet.config.hidden_size +
            text_model_bert.config.hidden_size,
            num_classes
        )

    def forward(self, img, txt_xlnet, txt_bert):
        img_feat = self.image_model(img)
        txt_feat_xlnet = self.text_model_xlnet(**txt_xlnet).last_hidden_state.mean(dim=1)
        txt_feat_bert = self.text_model_bert(**txt_bert).last_hidden_state.mean(dim=1)

        combined_feat = torch.cat((img_feat, txt_feat_xlnet, txt_feat_bert), dim=1)
        return self.fc(combined_feat)
    
