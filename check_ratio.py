import os
from collections import Counter
from dataset import ExamCheatingDataset

def main():
    print("=== Training Set Ratio ===")
    train_ds = ExamCheatingDataset(
        feature_root="features",
        crop_root="crop",
        split="train",
        verbose=True
    )
    
    print("\n=== Valid Set Ratio ===")
    test_ds = ExamCheatingDataset(
        feature_root="features",
        crop_root="crop",
        split="valid",
        verbose=True
    )
    
    def print_ratio(ds, name):
        labels = [s[1] for s in ds.samples]
        counts = Counter(labels)
        
        # Based on load_per_student_labels:
        # return 1 if n_cheating > 0 else 0
        # So 1 = Cheating, 0 = Not Cheating
        cheating = counts.get(1, 0)
        not_cheating = counts.get(0, 0)
        total = cheating + not_cheating
        
        if total == 0:
            print(f"\n{name} Dataset Statistics:")
            print("No samples found.")
            return
            
        print(f"\n{name} Dataset Statistics:")
        print(f"Total samples: {total}")
        print(f"Cheating: {cheating} ({cheating/total*100:.2f}%)")
        print(f"Not Cheating: {not_cheating} ({not_cheating/total*100:.2f}%)")
        
        if not_cheating > 0:
            ratio = cheating / not_cheating
            print(f"Ratio (Cheating to Not Cheating): {ratio:.2f} : 1")
        elif cheating > 0:
            print(f"Ratio (Cheating to Not Cheating): All Cheating")
            
    print_ratio(train_ds, "Training")
    print_ratio(test_ds, "Valid")

if __name__ == "__main__":
    main()
