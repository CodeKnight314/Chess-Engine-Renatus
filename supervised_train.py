import argparse
from model import RenatusV2
import torch.optim as optim 
import torch 
import os
from tqdm import tqdm
import torch.nn as nn
from dataset import get_dataloader
    
def supervised_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = os.path.join(args.path, "saved_weights/")
    os.makedirs(output_path, exist_ok=True)
    
    model = RenatusV2(27, 19).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()    
    
    train_dl = get_dataloader(args.file_path, 32)
    
    for epoch in range(args.epochs):
        model.train() 
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for i, data in enumerate(tqdm(train_dl, desc=f"[Training Renatus][{epoch+1}/{args.epochs}]")):
            opt.zero_grad()
            current_state_tensor, policy_index, value_tensor = data
            
            # Move tensors to device
            current_state_tensor = current_state_tensor.to(device)
            policy_index = policy_index.to(device)
            value_tensor = value_tensor.to(device)
            
            # Forward pass
            pred_policy, pred_value = model(current_state_tensor)
            
            # Compute losses
            policy_loss = policy_loss_fn(pred_policy, policy_index)
            value_loss = value_loss_fn(pred_value, value_tensor)
            
            # Combine losses
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            opt.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Print epoch statistics
        avg_policy_loss = total_policy_loss / len(train_dl)
        avg_value_loss = total_value_loss / len(train_dl)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Policy Loss: {avg_policy_loss:.4f}, Average Value Loss: {avg_value_loss:.4f}")
        
        # Save model weights
        torch.save(model.state_dict(), os.path.join(output_path, f"renatusv2_epoch{epoch+1}.pth"))
    
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the PGN file.")
    parser.add_argument("--path", type=str, required=True, help="Path to save model weights.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    
    args = parser.parse_args()
    supervised_train(args)