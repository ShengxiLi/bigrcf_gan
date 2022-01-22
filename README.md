# BigRCF-GAN
This repo wraps BigGAN to implement the proposed RCF-GAN. The detailed introduction of the RCF-GAN can be found at here https://github.com/ShengxiLi/rcf_gan.

## How To Install
Follow the steps in the original BigGAN-Pytorch repo: https://github.com/ajbrock/BigGAN-PyTorch

## How To Use
We have provided example scripts. Briefly speaking, the BigRCF-GAN method is enabled by mainly specifying ```--model RcfGAN```, to use the RCF-GAN model, together with ```--which_train_fn RCFGAN```, to enable the corresponding training strategy. Otherwise, you are training your models for unconditional BigGAN generation. The t-net is enabled by default and can be disabled by ```--t_sigma_num 0```. Please also note that the RCF-GAN, in its current form, only supports unconditional generation, so that a self-modulation technique is employed by specifying ```--unconditional```.

## Special Thanks
* The basis of the BigRCF-GAN is built upon the official BigGAN-Pytorch at https://github.com/ajbrock/BigGAN-PyTorch.
* The implementation of self-modulation is adopted from https://github.com/boschresearch/unetgan.

