fn main() { let cmd = serde_json::from_str::<eaRS::WebSocketCommand>(r#"{"Restart":{}}"#).unwrap(); println\!("{:?}", cmd); }
