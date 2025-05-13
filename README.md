
<h1 align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1E90FF,100:00BFFF&height=115&section=header"/>
</h1>

# IM2ELEVATION : Building Height Estimation from Single-View Aerial Imagery

## Table of contents
- [Data](#data)
- [Case of use](#case-of-use)
- [Results](#results)
- [Source](#source)

## Data

You can download the trained weight from below link:

- [Download the trained weight here](https://drive.google.com/file/d/1KZ50MQY5Fof8SAoJN34bxnk7mfou4frF/view?usp=sharing)


Please find the link below to download the OSI dataset.

- [Download the OSI dataset](https://drive.google.com/drive/folders/14sBkjeYY7R1S9NzWI5fGLX8XTuc8puHy?usp=sharing)

## Case of use
First place yourself in the cloned repository and build the docker image:
```
    docker build -t imele-image .
```

And you can now start the bash env to use python:
```
    docker run -it --rm -v path/to/data:/data -v ./:/work  imele-image bash
```
As an example you can use **test.py** to produce evaluation on a _tiff_ file and compared it to ground truth.
```
    python test.py --model '/path/to/model/.tar' --csv '/path/to/csv' --out '/path/to/out/directory'
```
## Results
At the moment, resutls are still not conclusive:


<p align="center">
    <table>
        <thead>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Loss</td>
                <td>-1.6579</td>
            </tr>
            <tr>
                <td>MSE</td>
                <td>291.8920</td>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>17.0848</td>
            </tr>
            <tr>
                <td>MAE</td>
                <td>17.0827</td>
            </tr>
            <tr>
                <td>SSIM</td>
                <td>-0.6360</td>
            </tr>
        </tbody>
    </table>
</p>

## Source
This package relates to the work presented in the following publication:

```
IM2ELEVATION: Building Height Estimation from Single-View Aerial Imagery. 
Liu, C.-J.; Krylov, V.A.; Kane, P.; Kavanagh, G.; Dahyot, R. 
Remote Sens. 2020, 12, 2719.
DOI:10.3390/rs12172719

```

Please cite this  journal paper (that is available Open Access [PDF](https://www.mdpi.com/2072-4292/12/17/2719/pdf)) when using this code or the dataset


<h1 align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1E90FF,100:00BFFF&height=115&reversal=true&section=footer"/>
</h1>