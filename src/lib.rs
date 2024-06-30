use std::fs;
use std::io::Write;
use std::sync::Mutex;
use rustler::JobSpawner;
use opencv::imgcodecs::{imencode,
//                        ImwriteFlags
};

use opencv::{
    videoio,
    prelude::*,
    core,
    core::{Mat, CV_8UC1, CV_8UC3},
    dnn,
    imgproc,
    types,
    videoio::{VideoCaptureTrait, VideoWriter, VideoWriterTrait},
};

use rustler::{
    Env, Term, NifResult, resource, ResourceArc, thread::ThreadSpawner
};


mod atoms {
    rustler::atoms! {
        ok,
        error,
    }
}

struct Lawi (Mutex<Mat>);

fn on_load(env: Env, _info: Term) -> bool {
    resource!(Lawi, env);
    true
}

#[rustler::nif(schedule = "DirtyIo")]
fn start_detection<'a>(env: Env<'a>) -> NifResult<ResourceArc<Lawi>>
{
    let mat = Mat::default();
    let struc: ResourceArc<Lawi> = ResourceArc::new(Lawi(Mutex::new(mat)));

    let struc_clone = ResourceArc::clone(&struc);
    ThreadSpawner::spawn(move || {
       let _ =  def(&struc_clone);
    });
    Ok(struc)
}



fn def<'a>(stream: &ResourceArc<Lawi>) -> Result<(), Box< dyn std::error::Error>> {
    let mut video_capture = videoio::VideoCapture::new(0, videoio::CAP_GSTREAMER)?;
    let mut net = dnn::read_net_from_darknet("blobs/yolov3.cfg", "blobs/yolov3.weights")?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    let classes = read_file("blobs/coco.names")?;

    process_video(&mut video_capture, &mut net, &classes, "output.avi", stream)?;
    Ok(())
}

fn process_video(
    video_capture: &mut videoio::VideoCapture,
    net: &mut dnn::Net,
    classes: &types::VectorOfString,
    output_file: &str,
    stream: &ResourceArc<Lawi>
) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = Mat::default();
    let mut blob = Mat::default();

    let conf_threshold = 0.5_f32;
    let nms_threshold = 0.4_f32;
    let inp_width = 416;
    let inp_height = 416;

    let frame_width = video_capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = video_capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = video_capture.get(videoio::CAP_PROP_FPS)?;
    println!("frame_width is {}, frame height is {}, fps is {}", &frame_width, & frame_height, & fps );

    let mut video_writer = VideoWriter::new(
        output_file,
        videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?,
        30_f64,
        core::Size::new(frame_width, frame_height),
        true,
    )?;

    while video_capture.read(&mut img)? {
        std::thread::sleep(std::time::Duration::from_millis(1000));
        let img_width = img.cols();
        let img_height = img.rows();
        if img.size()?.width == 0 {
            break;
        }

        dnn::blob_from_image_to(
            &img,
            &mut blob,
            1. / 255.,
            core::Size::new(inp_width, inp_height),
            core::Scalar::new(0., 0., 0., 0.),
            false,
            false,
            core::CV_32F,
        )?;

        let names = get_output_names(&net)?;
        let mut net_output   = types::VectorOfMat::new();
        net.set_input(&blob, "", 1.0, core::Scalar::new(0., 0., 0., 0.))?;
        net.forward(&mut net_output, &names)?;

        let mut class_ids = types::VectorOfi32::new();
        let mut confidences = types::VectorOff32::new();
        let mut boxes = types::VectorOfRect::new();

        for matrix in net_output.iter() {
            for row in 0..matrix.rows() {
                let data = matrix.at_row::<f32>(row)?;
                let scores = &data[5..];
                let (class_id_point, confidence) = scores.iter().enumerate()
                    .fold((0, 0.0), |(max_i, max_val), (i, &val)| {
                        if val > max_val {
                            (i, val)
                        } else {
                            (max_i, max_val)
                        }
                    });

                if confidence > conf_threshold {
                    let center_x = (data[0] * img_width as f32) as i32;
                    let center_y = (data[1] * img_height as f32) as i32;
                    let width = (data[2] * img_width as f32) as i32;
                    let height = (data[3] * img_height as f32) as i32;
                    let left = center_x - width / 2;
                    let top = center_y - height / 2;

                    class_ids.push(class_id_point as i32);
                    confidences.push(confidence);
                    boxes.push(core::Rect::new(left, top, width, height));
                }
            }
        }

        let mut indices = types::VectorOfi32::new();
        dnn::nms_boxes(&boxes, &confidences, conf_threshold, nms_threshold, &mut indices, 1., 0)?;

        for num in indices.iter() {
            let bbox = boxes.get(num as usize)?;
            let label = classes.get(class_ids.get(num as usize)? as usize)?;

            imgproc::rectangle(&mut img, bbox, core::Scalar::new(255., 18., 50., 0.0), 2, imgproc::LINE_8, 0)?;
            imgproc::put_text(
                &mut img,
                &label,
                core::Point::new(bbox.x, bbox.y),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.75,
                core::Scalar::new(0., 0., 0., 0.),
                1,
                imgproc::LINE_8,
                false,
            )?;
        }
        let mut akbar_zab  = stream.0.lock().unwrap();
        *akbar_zab =  img.clone();
    }

    video_writer.release()?;
    Ok(())
}

fn get_output_names(net: &dnn::Net) -> Result<types::VectorOfString, Box<dyn std::error::Error>> {
    let layers = net.get_unconnected_out_layers()?;
    let layer_names = net.get_layer_names()?;
    Ok(layers.iter().enumerate().fold(types::VectorOfString::new(), |mut names, (i, _)| {
        names.insert(i, &layer_names.get((layers.get(i).unwrap() - 1) as usize).expect("No such value.")).expect("Failed inserting value.");
        names
    }))
}

fn read_file(file_name: &str) -> Result<types::VectorOfString, Box<dyn std::error::Error>> {
    Ok(fs::read_to_string(file_name)?.split_whitespace().map(|name| name.into()).collect())
}


rustler::init!("Elixir.YoloNif", [start_detection, read_chunk], load= on_load);


#[rustler::nif]
fn read_chunk(env: Env, stream: ResourceArc<Lawi>) -> NifResult<Vec<u8>> {
    let stream = stream.0.lock().unwrap();
     let mat = &(*stream);
    let mut output_image = opencv::core::Vector::new();
    let params = opencv::types::VectorOfi32::new();


    imencode(".jpg", mat, &mut output_image, &params).unwrap();

    let _ = save_binary_to_file("output_test.jpg",&output_image.as_slice());
    Ok(output_image.into())
}

fn save_binary_to_file(file_path: &str, data: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(file_path)?;
    file.write_all(data)?;
    Ok(())
}

fn mat_to_binary_image(mat: &Mat) -> Result<Vec<u8>, opencv::Error> {

    let mat_continuous = if mat.is_continuous() {
        mat.clone()
    } else {
        let mut continuous = Mat::default();
        mat.copy_to(&mut continuous)?;
        continuous
    };

    // Check the type of Mat and convert accordingly
    let mat_type = mat_continuous.typ();
    let binary_data = match mat_type {
        CV_8UC1 | CV_8UC3 => {
            // Get the raw data as a slice
            let size = mat_continuous.total() * mat_continuous.elem_size()?;
            let slice = unsafe {
                std::slice::from_raw_parts(mat_continuous.ptr(0)? as *const u8, size)
            };
            slice.to_vec()
        },
        _ => return Err(opencv::Error::new(0, "Unsupported Mat type")),
    };

    Ok(binary_data)
}
