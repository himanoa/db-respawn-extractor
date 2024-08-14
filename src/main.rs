use image::{DynamicImage, GenericImage, ImageReader, Rgba};
use opencv::core::{in_range, Mat, MatTrait, Point, Scalar, Vec3b};
use opencv::imgproc::{bounding_rect, find_contours, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL};
use opencv::prelude::*;

use std::env;
use std::process::exit;

fn main() {
    let args = env::args().collect::<Vec<String>>();
    let file_name = args.get(1);
    let (file_name, image) = match file_name {
        Some(name) => {
            (name, ImageReader::open(name).unwrap().decode().unwrap())
        }
        None => {
            println!("missing file name");
            exit(1);
        }
    };
    let map_image = convert_image_to_mat(pre_process(image, file_name.to_string(), true));
    let cordinate = get_cordinate(map_image);
    let interactive_map_cordinate = InteractiveMapCordinate::from(cordinate);
    println!("player_marker_cordinate: {},{}", cordinate.0, cordinate.1);
    println!("player_marker_cordinate(interactive_map_feedback_format): {},{}", interactive_map_cordinate.0, interactive_map_cordinate.1);
}

struct Rgb(usize, usize, usize);
struct Bgr(f64, f64, f64);

impl Rgb {
    fn to_bgr(&self) -> Bgr {
        Bgr(self.2 as f64, self.1 as f64, self.0 as f64)
    }
}

fn pre_process(mut image: DynamicImage, name: String, is_dump_progress: bool) -> DynamicImage
{
    let black = Rgba([0,0,0,255]);
    let corner_rect_size = (45,32);
    let cropped_image = image.crop(1057, 102, 1724, 1727);
    if is_dump_progress {
        let _ = cropped_image.save(format!("{}_cropped.jpg", name));
    }
    let mut resized_image = cropped_image.resize(400, 400, image::imageops::FilterType::Lanczos3);
    if is_dump_progress {
        let _ = resized_image.save(format!("{}_resized.jpg", name));
    }

    // 四隅
    for x in 0..(corner_rect_size.0) {
        for y in 0..(corner_rect_size.1) {
            resized_image.put_pixel(x,y,black);
        }
    }

    for x in (resized_image.width() - corner_rect_size.0)..(resized_image.width()) {
        for y in 0..(corner_rect_size.1) {
            resized_image.put_pixel(x,y,black);
        }
    }

    for x in 0..(corner_rect_size.0) {
        for y in (resized_image.height() - corner_rect_size.1)..(resized_image.height()) {
            resized_image.put_pixel(x,y,black);
        }
    }

    for x in (resized_image.width() - corner_rect_size.0)..(resized_image.width()) {
        for y in (resized_image.height() - corner_rect_size.1)..(resized_image.height()) {
            resized_image.put_pixel(x,y,black);
        }
    }

    // 上部
    for x in 0..(resized_image.width()) {
        for y in 0..25 {
            resized_image.put_pixel(x,y,black);
        }
    }

    // 下部
    for x in 0..(resized_image.width()) {
        for y in (resized_image.height() - 30)..(resized_image.height()) {
            resized_image.put_pixel(x,y,black);
        }
    }


    // 左部
    for x in 0..10 {
        for y in 0..(resized_image.height()) {
            resized_image.put_pixel(x,y,black);
        }
    }

    // 右部
    for x in (resized_image.width() - 10)..(resized_image.width()) {
        for y in 0..(resized_image.height()) {
            resized_image.put_pixel(x,y,black);
        }
    }

    if is_dump_progress {
        let _ = resized_image.save(format!("{}_filled_corners.jpg", name));
    }
    resized_image
}


fn convert_image_to_mat(image: DynamicImage) -> Mat {
    let rgb_image = image.to_rgb8();

    let (width, height) = rgb_image.dimensions();
    let mut mat = Mat::zeros(height as i32, width as i32, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb_image.get_pixel(x, y);
            mat.at_2d_mut::<Vec3b>(y as i32, x as i32).unwrap().0 = [pixel[2], pixel[1], pixel[0]];
        }
    }

    mat
}

fn get_cordinate(mat: Mat) -> Cordinate {
    let red = Rgb(157, 45, 41);
    let red_bgr = red.to_bgr();
    const RANGE: f64 = 30.0;
    let lower_red = Scalar::new(red_bgr.0 - RANGE, red_bgr.1 - RANGE,  red_bgr.2 - RANGE, 0.0);
    let upper_red = Scalar::new(red_bgr.0 + RANGE, red_bgr.1 + RANGE,  red_bgr.2 + RANGE, 0.0);

    let mut mask = Mat::default();
    in_range(&mat, &lower_red, &upper_red, &mut mask).unwrap();
    let mut contours: opencv::core::Vector<opencv::core::Vector<opencv::core::Point>> = opencv::core::Vector::new();

    find_contours(
        &mask,
        &mut contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    ).unwrap();

    let targets = contours.iter()
        .map(|contour| {
            bounding_rect(&contour).expect("failed bounding_rect error")
        })
        .filter(|rect| {
            !(rect.width == 1 && rect.height == 1) && !(rect.width == 5 && rect.height == 9)
        }).collect::<Vec<_>>();

    let rect = targets[0];
    Cordinate(rect.x + rect.width / 2, rect.y + rect.height / 2)
}


#[derive(Debug, Clone, Copy)]
struct Cordinate(i32, i32);

#[derive(Debug, Clone, Copy)]
struct InteractiveMapCordinate(i32, i32);

impl From<Cordinate> for InteractiveMapCordinate {
    fn from(value: Cordinate) -> Self {
        InteractiveMapCordinate(value.0, 400 - value.1)
    }
}

