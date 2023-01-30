# %%
import numpy as np
import  matplotlib.pyplot as plt
import vedo


rng = np.random.default_rng(seed=42)
# %%
#vedo.Cone().show(axes=1).close()


#create random array of arrays with size 256x256
w, h = 256, 256
paddingWidth = 5
numberOfImages = 256  # must be devisible by rowbreakAfter
rowbreakAfter = 16

data = rng.integers(0, 256, size=(numberOfImages, h, w), dtype=np.uint8)

#do not pad allong fist axis, add "paddingWidth" padding to beginning and end of second and third axis  
npad = ((0, 0), (paddingWidth, paddingWidth), (paddingWidth, paddingWidth))
padded_data = np.pad(data, pad_width=npad, mode='constant', constant_values=255)

rows = []
for i in range(0, numberOfImages, rowbreakAfter):
    #concatenate arrays along "w" axis
    row = np.concatenate(padded_data[i:i+rowbreakAfter], axis=1)
    rows.append(row)

#concatenate arrays along "h" axis
conc_data = np.concatenate(rows, axis=0)


# plt.figure()
# plt.imshow(conc_data, cmap='gray')

vedo.Picture(conc_data).show(axes=1).close()
