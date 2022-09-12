#!/bin/bash

p='../train.py'
g='0'
dr=$1

## P2R
python ${p} \
--source P \
--target R \
--dataset_root ${dr} \
--epochs 60 \
--pretrain_epochs 10 \
--margin 0.05 \
--pre_reg \
--train_margin \
--train_reg \
--multi_unk \
--corresp \
--revise_prototype post \
--expname exp1 \
--test_name AMS \
--seed 2022 \
--gpuid ${g} \
--log_mode \
--save_model &&

## R2P
python ${p} \
--source R \
--target P \
--dataset_root ${dr} \
--epochs 40 \
--pretrain_epochs 10 \
--margin 0.05 \
--pre_reg \
--train_margin \
--train_reg \
--multi_unk \
--corresp \
--revise_prototype post \
--lr 0.5e-3 \
--expname exp1 \
--test_name AMS \
--seed 2022 \
--gpuid ${g} \
--log_mode \
--save_model &&

## R2A
python ${p} \
--source R \
--target A \
--dataset_root ${dr} \
--epochs 20 \
--pretrain_epochs 10 \
--margin 0.05 \
--pre_reg \
--train_margin \
--train_reg \
--multi_unk \
--corresp \
--revise_prototype post \
--expname exp1 \
--test_name AMS \
--seed 2022 \
--gpuid ${g} \
--log_mode \
--save_model &&

## A2R
python ${p} \
--source A \
--target R \
--dataset_root ${dr} \
--epochs 40 \
--pretrain_epochs 10 \
--margin 0.05 \
--pre_reg \
--train_margin \
--train_reg \
--multi_unk \
--corresp \
--revise_prototype post \
--lr 0.5e-3 \
--expname exp1 \
--test_name AMS \
--seed 2022 \
--gpuid ${g} \
--log_mode \
--save_model &&

## P2A
python ${p} \
--source P \
--target A \
--dataset_root ${dr} \
--epochs 60 \
--pretrain_epochs 10 \
--margin 0.05 \
--pre_reg \
--train_margin \
--train_reg \
--multi_unk \
--corresp \
--revise_prototype post \
--expname exp1 \
--test_name AMS \
--seed 2022 \
--gpuid ${g} \
--log_mode \
--save_model &&

## A2P
python ${p} \
--source A \
--target P \
--dataset_root ${dr} \
--epochs 40 \
--pretrain_epochs 10 \
--margin 0.05 \
--pre_reg \
--train_margin \
--train_reg \
--multi_unk \
--corresp \
--revise_prototype post \
--lr 0.5e-3 \
--expname exp1 \
--test_name AMS \
--seed 2022 \
--gpuid ${g} \
--log_mode \
--save_model &&

## I2A
python ${p} \
--source 3D2 \
--target A \
--dataset_root ${dr} \
--epochs 100 \
--pretrain_epochs 10 \
--margin 0.05 \
--pre_reg \
--train_margin \
--train_reg \
--multi_unk \
--corresp \
--revise_prototype post \
--expname exp1 \
--test_name AMS \
--seed 2022 \
--gpuid ${g} \
--log_mode \
--save_model
