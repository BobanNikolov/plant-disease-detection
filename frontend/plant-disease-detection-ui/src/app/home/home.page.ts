import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { Camera, CameraResultType, CameraSource } from '@capacitor/camera';
import { Geolocation } from '@capacitor/geolocation';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {

  imageUrl: string | undefined;
  prediction: string | undefined;

  constructor(private http: HttpClient) {}

  async takePicture() {
    const image = await Camera.getPhoto({
      quality: 90,
      allowEditing: false,
      resultType: CameraResultType.Uri,
      source: CameraSource.Camera
    });

    this.imageUrl = image.webPath;

    const imageBlob = await this.fetchImageAsBlob(image.webPath as string);
    const resizedImageBlob = await this.resizeImage(imageBlob, 225, 225);

    const formData = new FormData();
    formData.append('image', resizedImageBlob, 'image.jpg');

    this.http.post<{ prediction: string }>(environment.backendUrl + '/predict', formData).subscribe(async result => {
      this.prediction = result.prediction;
      const coordinates = await Geolocation.getCurrentPosition();

      const measurementData = {
        lon: coordinates.coords.longitude.toString(),
        lat: coordinates.coords.latitude.toString(),
        time_of_measurement: new Date().toISOString(),  
        predicted_result: this.prediction
      };
      console.log(measurementData)

      this.http.post(environment.backendUrl + '/measurement', measurementData).subscribe();
    });
  }

  resizeImage(blob: Blob, width: number, height: number): Promise<Blob> {
    return new Promise(resolve => {
      const img = document.createElement('img');
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      img.onload = () => {
        canvas.width = width;
        canvas.height = height;
        ctx?.drawImage(img, 0, 0, width, height);
        canvas.toBlob(blob => {
          if (blob) {
            resolve(blob);
          } else {
            console.log('Image resizing failed.');
          }
        }, 'image/jpeg');
      };

      img.src = URL.createObjectURL(blob);
    });
  }

  
  async fetchImageAsBlob(imageUrl: string): Promise<Blob> {
    const response = await fetch(imageUrl);
    return await response.blob();
  }

}
