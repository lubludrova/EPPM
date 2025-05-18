import os
import tempfile
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from src.utils.utils import Utils
# from ray import tune
# import wandb
import datetime


class Trainer:
    def __init__(self, generator, val_generator, model, attribute_dict, log, model_name, n_train_samples, architecture, test_generator, search_mode=False, optimization_mode="min", batch_size=32):
        self.epoch = 3
        self.generator = generator
        self.model = model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.val_generator = val_generator
        self.attribute_dict = attribute_dict
        self.log = log + "_" + architecture
        self.model_name = model_name
        self.search_mode = search_mode
        self.optimization_mode = optimization_mode
        self.n_train_samples = n_train_samples
        self.batch_size = batch_size

        self.activity_loss = torch.nn.CrossEntropyLoss()
        self.timestamp_loss = torch.nn.L1Loss()
        self.attribute_losses = []
        self.test_generator = test_generator

        for attribute in list(attribute_dict.keys()):
            self.attribute_losses.append(torch.nn.CrossEntropyLoss(reduction="none"))

        # # wandb.login(key="b71ef916f7bf3610b29fa45f1e7d1f0af02dc03e")
        # # wandb.init(project="master")
        # # wandb.run.name = architecture + "_" + self.log +  "_" + str(datetime.datetime.now())
        # # wandb.watch(self.model, log="all")

    def train(self):
        # Disable tqdm bar
        epoch_without_improving = 0
        if self.optimization_mode == "min":
            best_metric = 99999
        else:
            best_metric = 0
        EARLY_STOPPING = 10

        #optimizer = self.model.optimizer(self.model.parameters(), lr=0.001)
        optimizer = self.model.optimizer(self.model.parameters(), lr=self.model.lr, weight_decay=0.0001)
        optimizer_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, last_epoch=-1, eta_min=0.0001)
        total = int(np.floor(self.n_train_samples / self.batch_size))

        Utils._make_dir_if_not_exists("./data/tensorboard/" + self.log)



        # if not self.search_mode:
        #     print("Starting model state dict: ", self.model.state_dict())

        for epoch, i in enumerate(range(self.epoch), 1):
            epoch_loss = []
            act_loss = []
            time_loss = []
            activity_predictions = []
            activity_real = []

            self.model.train()
            if not self.search_mode:
                pbar = tqdm(self.generator, total=total)
                pbar.set_description("Epoch " + str(epoch) + " of " + str(self.epoch))
            else:
                pbar = self.generator

            iteration = 0
            for step, (X, y, max_batch_length, last_places_activated_batch) in enumerate(pbar):
                outputs = self.model(X[0], X[1], max_batch_length, last_places_activated_batch, X[2])
                unique_labels, counts = np.unique(y, return_counts=True)
                # print("🎯 Классы в y_train:", dict(zip(unique_labels, counts)))

                act_output = outputs[0].cpu()
                real_act = torch.from_numpy(y[0])


                loss_activities = self.activity_loss(act_output, real_act).mean()

                """
                loss_timestamp = self.timestamp_loss(outputs[1].cpu(), torch.from_numpy(y[1]))
                losses = [loss_activities, loss_timestamp]
                for i, attribute in enumerate(list(self.attribute_dict.keys())):
                    loss_attribute = self.attribute_losses[i](outputs[i+2].cpu(), torch.from_numpy(y[i+2]))
                    losses.append(loss_attribute)
                """

                losses = [loss_activities]
                total_loss = sum(losses)

                optimizer.zero_grad()
                total_loss.backward()

                #torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)

                optimizer.step()
                optimizer_sch.step()

                epoch_loss.append(total_loss.item())
                act_loss.append(loss_activities.item())
                #time_loss.append(loss_timestamp.item())

                next_activity_probs = torch.softmax(act_output.detach(), dim=-1).numpy()
                pred_argmax = np.argmax(next_activity_probs, axis=-1)
                for real, pred in zip(real_act, pred_argmax):
                    activity_real.append(real)
                    activity_predictions.append(pred)

                #pbar.set_description("Loss: " + str(np.mean(epoch_loss)))
                train_acc = accuracy_score(activity_real, activity_predictions)


                if step % 50 == 0 or step == total - 1:
                    print("[DEBUG] Unique labels in y:", torch.unique(real_act, return_counts=True))
                    print(f"[Train] Epoch {epoch}, Step {step}/{total} — Loss: {total_loss.item():.6f}, Acc: {train_acc:.4f}")

                    if total_loss.item() < 1e-5:
                        print("\u26a1 Very low loss detected!")
                        print("Predicted class dist:", np.bincount(pred_argmax))
                        print("True class dist:", np.bincount(real_act.numpy()))

                if not self.search_mode:
                    pbar.set_postfix({
                        "loss" : np.mean(epoch_loss),
                        "act_loss" : np.mean(act_loss),
                        #"time_loss" : np.mean(time_loss),
                        "activity_acc" : train_acc
                    })


                #grads.append(self.model.ggrnn.GGRNN_1.gru_cell.conv_z_1.kernel.grad)
                iteration += 1

            print(f"\n🔹 End of Epoch {epoch}: Train Loss = {np.mean(epoch_loss):.6f}, Train Accuracy = {train_acc:.4f}")
            print("Sample prediction vs real:", pred_argmax[-5:], [r.item() for r in real_act[-5:]])

            # # wandb.log({"train_acc" : train_acc, "epoch" : epoch, "train_loss" : np.mean(epoch_loss)}, step=epoch)

            print("\n🔍 Starting validation...")
            # Calculate validation loss
            self.model.eval()
            with torch.no_grad():
                val_pbar = tqdm(self.val_generator.get(), total=len(self.val_generator))
                losses = []
                y_pred = []
                y_true = []
                for X, y, max_batch_length, last_places_activated_batch in val_pbar:
                    preds = self.model(X[0], X[1], max_batch_length, last_places_activated_batch, X[2])
                    preds_output = preds[0].cpu()
                    #real_act = torch.from_numpy(y[0])
                    real_act = torch.from_numpy(y[0])

                    loss_activities = self.activity_loss(preds_output, real_act)
                    losses.append(loss_activities.item())

                    pred_argmax = np.argmax(torch.softmax(preds_output.detach(), dim=-1).numpy(), axis=-1)
                    for real, pred in zip(real_act, pred_argmax):
                        y_true.append(real)
                        y_pred.append(pred)

                mean_loss = np.mean(losses)
                mean_acc = accuracy_score(y_true, y_pred)

                labels_sorted = sorted(set(y_true) | set(y_pred))
                cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

                cm_df = pd.DataFrame(
                    cm,
                    index=[f"T:{l}" for l in labels_sorted],
                    columns=[f"P:{l}" for l in labels_sorted],
                )

                print("[DEBUG] preds type:", type(preds))  # Покажет тип preds (list/tuple)
                print("[DEBUG] preds length:", len(preds))  # Сколько элементов в preds
                print("[DEBUG] preds[0].shape:", preds[0].shape)  # Форма первого выхода модели (тензор активности)
                if len(preds) > 1:
                    print("[DEBUG] preds[1].shape:", preds[1].shape)  # Если есть другие выходы (например, места)

                # pred_classes уже считается от preds_output (preds[0]), поэтому оставляем как есть
                pred_classes = torch.argmax(preds_output.detach(), dim=-1)  # Используем preds_output вместо preds
                print("[DEBUG] pred_classes shape:", pred_classes.shape)
                print("[DEBUG] pred_classes:", pred_classes[:5])
                print("[DEBUG] y:", y[:5])

                if not self.search_mode:
                    print(f"Total val samples: {len(y_true)}")
                    print(f"Val Loss Mean: {mean_loss:.6f}, Val Accuracy: {mean_acc:.4f}")
                    print("Sample VAL prediction vs real:", y_pred[-5:], y_true[-5:])
                    # print("\n🧩 Validation Confusion Matrix:")
                    # print(cm_df.to_string())

                if self.optimization_mode == "min":
                    if mean_loss <= best_metric:
                        print("Best loss improved from " + str(best_metric) + " to " + str(mean_loss) + ". Saving model to disk")
                        best_metric = mean_loss
                        epoch_without_improving = 0
                        save_path = "./data/torch_models/" + self.log
                        Utils._make_dir_if_not_exists(save_path)
                        # torch.save(self.model.state_dict(), save_path + "/" + self.model_name)
                        torch.save({
                            "ggrnn": self.model.state_dict(),  # как раньше
                            "head": self.model.learning_model.head.state_dict()
                        }, save_path + "/" + self.model_name)
                    else:
                        epoch_without_improving += 1
                else:
                    if best_metric <= mean_acc:
                        print("Best acc improved from " + str(best_metric) + " to " + str(mean_acc) + ". Saving model to disk")
                        best_metric = mean_acc
                        epoch_without_improving = 0
                        save_path = "./data/torch_models/" + self.log
                        Utils._make_dir_if_not_exists(save_path)
                        # torch.save(self.model.state_dict(), save_path + "/" + self.model_name)
                        torch.save({
                            "ggrnn": self.model.state_dict(),  # как раньше
                            "head": self.model.learning_model.head.state_dict()
                        }, save_path + "/" + self.model_name)
                    else:
                        epoch_without_improving += 1


                if not self.search_mode:
                    if epoch_without_improving >= EARLY_STOPPING:
                        if not self.search_mode:
                            print("Early stopping after ", EARLY_STOPPING, "epochs ")
                        break

            print("\n🎯 Evaluating on test set...")
            if not self.search_mode:
                with torch.no_grad():
                    if not self.search_mode:
                        test_pbar = tqdm(self.test_generator.get(), total=len(self.test_generator))
                    else:
                        test_pbar = self.test_generator.get()
                    #test_pbar = self.test_generator.get()
                    losses = []
                    y_pred = []
                    y_true = []
                    for X, y, max_batch_length, last_places_activated_batch in test_pbar:
                        preds = self.model(X[0], X[1], max_batch_length, last_places_activated_batch, X[2])
                        preds_output = preds[0].cpu()
                        #real_act = torch.from_numpy(y[0])
                        real_act = torch.from_numpy(y[0])

                        loss_activities = self.activity_loss(preds_output, real_act)
                        losses.append(loss_activities.item())

                        pred_argmax = np.argmax(torch.softmax(preds_output.detach(), dim=-1).numpy(), axis=-1)
                        for real, pred in zip(real_act, pred_argmax):
                            y_true.append(real)
                            y_pred.append(pred)

                    mean_acc = accuracy_score(y_true, y_pred)
                    print(f"Test Loss Mean: {np.mean(losses):.6f}, Test Accuracy: {mean_acc:.4f}")
                    print("Sample TEST prediction vs real:", y_pred[-5:], y_true[-5:])

                    labels_sorted_test = sorted(set(y_true) | set(y_pred))
                    cm_test = confusion_matrix(y_true, y_pred, labels=labels_sorted_test)
                    cm_test_df = pd.DataFrame(
                        cm_test,
                        index=[f"T:{l}" for l in labels_sorted_test],
                        columns=[f"P:{l}" for l in labels_sorted_test],
                    )
                    # print("\n🧰 Confusion Matrix (TEST):")
                    # print(cm_test_df.to_string())


    def plot_gradients(self, tensorboard_writer, model, epoch, iteration, max_iteration):
        # Plot gradients
        curr_iteration = (epoch-1) * max_iteration + iteration
        for name, param in self.model.named_parameters():
            #print("NAME: ", name)
            if "weight" in name or "kernel" in name:
                # Plot weight
                #print("Param: ", param)
                #print("Param grad: ", param.grad)
                tensorboard_writer.add_histogram(name,
                                                 param,
                                                 curr_iteration
                                                 )
                if param.grad is not None:
                    # Plot gradient
                    tensorboard_writer.add_histogram(name + ".grad",
                                                 param.grad,
                                                 curr_iteration
                                                 )

