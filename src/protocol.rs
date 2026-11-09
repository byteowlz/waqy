//! `wake-stream/1`: the process interface between a microphone owner (eaRS) and any detector.
//!
//! The host starts a detector command and talks to it over pipes. Nothing about the
//! model is known to the host; it only learns the phrases from the `ready` line.
//!
//! stdin, binary frames: `kind: u8`, `len: u32 little-endian`, `payload[len]`
//!   `A` audio: 16 kHz mono signed 16-bit little-endian PCM, any length
//!   `P` playback: one byte, 1 = host is playing audio (gate on), 0 = stopped
//!   `R` reset: no payload; forget audio context
//!   `Q` quit: no payload
//! stdout, JSON lines:
//!   {"type":"ready","protocol":"wake-stream/1","sample_rate":16000,"phrases":[...],"detector":"...","release":"..."}
//!   {"type":"detection","phrase":"hey_oqto","score":0.91,"at_sample":123456}
//!   {"type":"candidate", ...same fields...}   optional near miss below the threshold; hosts may log it, never act on it
//!   {"type":"error","message":"..."}
//! stderr: free-form logs.
//!
//! A detection is evidence that a phrase was probably spoken. The host decides what it
//! means; detectors never authenticate speakers or authorize actions.

use crate::detector::{Detection, SAMPLE_RATE, WakeDetector};
use anyhow::{Result, bail};
use serde::Serialize;
use std::io::{Read, Write};

pub const PROTOCOL: &str = "wake-stream/1";
const MAX_FRAME: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    Audio(Vec<f32>),
    Playback(bool),
    Reset,
    Quit,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event<'a> {
    Ready {
        protocol: &'a str,
        sample_rate: u32,
        phrases: Vec<String>,
        detector: &'a str,
        release: &'a str,
    },
    Detection(&'a Detection),
    Candidate(&'a Detection),
    Error {
        message: String,
    },
}

pub fn write_frame(out: &mut impl Write, frame: &Frame) -> std::io::Result<()> {
    match frame {
        Frame::Audio(pcm) => {
            out.write_all(b"A")?;
            out.write_all(&((pcm.len() * 2) as u32).to_le_bytes())?;
            let bytes: Vec<u8> = pcm
                .iter()
                .flat_map(|&x| ((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
                .collect();
            out.write_all(&bytes)
        }
        Frame::Playback(on) => {
            out.write_all(b"P")?;
            out.write_all(&1u32.to_le_bytes())?;
            out.write_all(&[*on as u8])
        }
        Frame::Reset => out.write_all(&[b'R', 0, 0, 0, 0]),
        Frame::Quit => out.write_all(&[b'Q', 0, 0, 0, 0]),
    }
}

/// Read one frame; `Ok(None)` at a clean end of input.
pub fn read_frame(input: &mut impl Read) -> Result<Option<Frame>> {
    let mut head = [0u8; 5];
    match input.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_le_bytes([head[1], head[2], head[3], head[4]]) as usize;
    if len > MAX_FRAME {
        bail!("frame of {len} bytes exceeds {MAX_FRAME}");
    }
    let mut payload = vec![0u8; len];
    input.read_exact(&mut payload)?;
    Ok(Some(match head[0] {
        b'A' => {
            if !len.is_multiple_of(2) {
                bail!("audio frame with odd byte count");
            }
            Frame::Audio(
                payload
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect(),
            )
        }
        b'P' if len == 1 => Frame::Playback(payload[0] != 0),
        b'R' if len == 0 => Frame::Reset,
        b'Q' if len == 0 => Frame::Quit,
        kind => bail!("unknown frame kind {kind:#x} with {len} bytes"),
    }))
}

fn emit(out: &mut impl Write, event: &Event) -> Result<()> {
    serde_json::to_writer(&mut *out, event)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

/// Serve one detector until stdin ends or `Q` arrives.
pub fn serve(
    detector: &mut dyn WakeDetector,
    name: &str,
    release: &str,
    input: &mut impl Read,
    output: &mut impl Write,
) -> Result<()> {
    emit(
        output,
        &Event::Ready {
            protocol: PROTOCOL,
            sample_rate: SAMPLE_RATE,
            phrases: detector.phrases(),
            detector: name,
            release,
        },
    )?;
    while let Some(frame) = read_frame(input)? {
        match frame {
            Frame::Audio(pcm) => match detector.push(&pcm) {
                Ok(found) => {
                    for d in &found {
                        emit(output, &Event::Detection(d))?;
                    }
                    for c in &detector.drain_candidates() {
                        emit(output, &Event::Candidate(c))?;
                    }
                }
                Err(e) => emit(
                    output,
                    &Event::Error {
                        message: e.to_string(),
                    },
                )?,
            },
            Frame::Playback(on) => detector.set_playback(on),
            Frame::Reset => detector.reset(),
            Frame::Quit => break,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo(u64);
    impl WakeDetector for Echo {
        fn phrases(&self) -> Vec<String> {
            vec!["test".into()]
        }
        fn push(&mut self, pcm: &[f32]) -> Result<Vec<Detection>> {
            self.0 += pcm.len() as u64;
            Ok(if pcm.iter().any(|&x| x > 0.5) {
                vec![Detection {
                    phrase: "test".into(),
                    score: 0.9,
                    at_sample: self.0,
                }]
            } else {
                vec![]
            })
        }
        fn set_playback(&mut self, _: bool) {}
        fn reset(&mut self) {
            self.0 = 0;
        }
    }

    #[test]
    fn frames_round_trip_and_serve_reports_detections() {
        let mut wire = Vec::new();
        write_frame(&mut wire, &Frame::Audio(vec![0.0; 100])).unwrap();
        write_frame(&mut wire, &Frame::Playback(true)).unwrap();
        write_frame(&mut wire, &Frame::Audio(vec![0.9; 10])).unwrap();
        write_frame(&mut wire, &Frame::Reset).unwrap();
        write_frame(&mut wire, &Frame::Quit).unwrap();
        write_frame(&mut wire, &Frame::Audio(vec![0.9; 10])).unwrap(); // after quit: ignored
        let mut out = Vec::new();
        serve(&mut Echo(0), "echo", "r0", &mut wire.as_slice(), &mut out).unwrap();
        let lines: Vec<serde_json::Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines[0]["type"], "ready");
        assert_eq!(lines[0]["protocol"], PROTOCOL);
        assert_eq!(lines[0]["phrases"][0], "test");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["type"], "detection");
        assert_eq!(lines[1]["at_sample"], 110);
    }

    #[test]
    fn rejects_unknown_and_oversized_frames() {
        assert!(read_frame(&mut [b'X', 0, 0, 0, 0].as_slice()).is_err());
        assert!(read_frame(&mut [b'A', 0xff, 0xff, 0xff, 0x7f].as_slice()).is_err());
        assert!(read_frame(&mut [b'A', 3, 0, 0, 0, 1, 2, 3].as_slice()).is_err());
        assert!(read_frame(&mut [].as_slice()).unwrap().is_none());
    }
}
