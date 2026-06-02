---------------------DOWN---------------
i 0
sample: torch.Size([1, 320, 64, 64])
---------------------DOWN---------------
i 1
sample: torch.Size([1, 320, 32, 32])
---------------------DOWN---------------
i 2
sample: torch.Size([1, 640, 16, 16])
---------------------DOWN---------------
i 3
sample: torch.Size([1, 1280, 8, 8])
---------------------UP---------------
j 0
sample: torch.Size([1, 1280, 8, 8])
self.hr_projs[j]: Conv2d(512, 1280, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 512, 16, 16])
up_block_additional_residuals: torch.Size([1, 1280, 16, 16])
---------------------UP---------------
j 1
sample: torch.Size([1, 1280, 16, 16])
self.hr_projs[j]: Conv2d(256, 1280, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 256, 32, 32])
up_block_additional_residuals: torch.Size([1, 1280, 32, 32])
---------------------UP---------------
j 2
sample: torch.Size([1, 1280, 32, 32])
self.hr_projs[j]: Conv2d(128, 640, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 128, 64, 64])
up_block_additional_residuals: torch.Size([1, 640, 64, 64])
---------------------UP---------------
j 3
sample: torch.Size([1, 640, 64, 64])
sr_dec: torch.Size([1, 4, 64, 64])
---------------------DOWN---------------
i 0
sample: torch.Size([1, 320, 64, 64])
---------------------DOWN---------------
i 1
sample: torch.Size([1, 320, 32, 32])
---------------------DOWN---------------
i 2
sample: torch.Size([1, 640, 16, 16])
---------------------DOWN---------------
i 3
sample: torch.Size([1, 1280, 8, 8])
---------------------UP---------------
j 0
sample: torch.Size([1, 1280, 8, 8])
self.hr_projs[j]: Conv2d(512, 1280, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 512, 16, 16])
up_block_additional_residuals: torch.Size([1, 1280, 16, 16])
---------------------UP---------------
j 1
sample: torch.Size([1, 1280, 16, 16])
self.hr_projs[j]: Conv2d(256, 1280, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 256, 32, 32])
up_block_additional_residuals: torch.Size([1, 1280, 32, 32])
---------------------UP---------------
j 2
sample: torch.Size([1, 1280, 32, 32])
self.hr_projs[j]: Conv2d(128, 640, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 128, 64, 64])
up_block_additional_residuals: torch.Size([1, 640, 64, 64])
---------------------UP---------------
j 3
sample: torch.Size([1, 640, 64, 64])
sr_dec: torch.Size([1, 4, 64, 64])
---------------------DOWN---------------
i 0
sample: torch.Size([1, 320, 64, 64])
---------------------DOWN---------------
i 1
sample: torch.Size([1, 320, 32, 32])
---------------------DOWN---------------
i 2
sample: torch.Size([1, 640, 16, 16])
---------------------DOWN---------------
i 3
sample: torch.Size([1, 1280, 8, 8])
---------------------UP---------------
j 0
sample: torch.Size([1, 1280, 8, 8])
self.hr_projs[j]: Conv2d(512, 1280, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 512, 16, 16])
up_block_additional_residuals: torch.Size([1, 1280, 16, 16])
---------------------UP---------------
j 1
sample: torch.Size([1, 1280, 16, 16])
self.hr_projs[j]: Conv2d(256, 1280, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 256, 32, 32])
up_block_additional_residuals: torch.Size([1, 1280, 32, 32])
---------------------UP---------------
j 2
sample: torch.Size([1, 1280, 32, 32])
self.hr_projs[j]: Conv2d(128, 640, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 128, 64, 64])
up_block_additional_residuals: torch.Size([1, 640, 64, 64])
---------------------UP---------------
j 3
sample: torch.Size([1, 640, 64, 64])
sr_dec: torch.Size([1, 4, 64, 64])
---------------------DOWN---------------
i 0
sample: torch.Size([1, 320, 64, 64])
---------------------DOWN---------------
i 1
sample: torch.Size([1, 320, 32, 32])
---------------------DOWN---------------
i 2
sample: torch.Size([1, 640, 16, 16])
---------------------DOWN---------------
i 3
sample: torch.Size([1, 1280, 8, 8])
---------------------UP---------------
j 0
sample: torch.Size([1, 1280, 8, 8])
self.hr_projs[j]: Conv2d(512, 1280, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 512, 16, 16])
up_block_additional_residuals: torch.Size([1, 1280, 16, 16])
---------------------UP---------------
j 1
sample: torch.Size([1, 1280, 16, 16])
self.hr_projs[j]: Conv2d(256, 1280, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 256, 32, 32])
up_block_additional_residuals: torch.Size([1, 1280, 32, 32])
---------------------UP---------------
j 2
sample: torch.Size([1, 1280, 32, 32])
self.hr_projs[j]: Conv2d(128, 640, kernel_size=(1, 1), stride=(1, 1))
up_block_additional_residuals[j]: torch.Size([1, 128, 64, 64])
up_block_additional_residuals: torch.Size([1, 640, 64, 64])
---------------------UP---------------
j 3
sample: torch.Size([1, 640, 64, 64])
sr_dec: torch.Size([1, 4, 64, 64])