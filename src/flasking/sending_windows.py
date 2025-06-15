import requests
import json
import time
import random

# Base URL of your Flask application
BASE_URL = "http://localhost:5000"


def send_data_to_window(window_num, data):
    """Send data to a specific window"""
    url = f"{BASE_URL}/api/window{window_num}"

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"✅ Data sent to Window {window_num}: {response.json()}")
        else:
            print(f"❌ Error sending to Window {window_num}: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error for Window {window_num}: {e}")


def main():
    print("🚀 Starting test client for Four Windows Flask App")
    print("📡 Sending test data to different windows...")

    # Sample nested JSON data for different windows
    test_data = {
        1: {
            "sensor_data": {
                "temperature": {
                    "value": 23.5,
                    "unit": "°C",
                    "readings": [22.1, 22.8, 23.5, 23.2, 23.9],
                    "metadata": {
                        "calibrated": True,
                        "last_calibration": "2024-01-10",
                        "accuracy": 0.1
                    }
                },
                "humidity": {
                    "value": 45.2,
                    "unit": "%",
                    "readings": [44.1, 45.0, 45.2, 45.8, 46.1],
                    "metadata": {
                        "calibrated": True,
                        "sensor_type": "DHT22"
                    }
                },
                "location": {
                    "room": "Server Room A",
                    "building": "Main Office",
                    "coordinates": {
                        "x": 12.5,
                        "y": 8.3,
                        "floor": 2
                    }
                }
            }
        },
        2: {
            "device_status": {
                "device_id": "DEV001",
                "name": "Main Server",
                "status": {
                    "online": True,
                    "last_ping": "2024-01-15T10:30:00Z",
                    "uptime": 8640000,
                    "health_check": {
                        "cpu": {"usage": 45.2, "temp": 62.1},
                        "memory": {"used": 8.2, "total": 16.0, "unit": "GB"},
                        "disk": {"used": 256, "total": 512, "unit": "GB"}
                    }
                },
                "network": {
                    "ip_address": "192.168.1.100",
                    "mac_address": "00:1B:44:11:3A:B7",
                    "ports": {
                        "ssh": 22,
                        "http": 80,
                        "https": 443
                    },
                    "bandwidth": {
                        "upload": 100.5,
                        "download": 250.8,
                        "unit": "Mbps"
                    }
                }
            }
        },
        3: {
            "user_activity": {
                "session": {
                    "user_id": "usr_12345",
                    "username": "john_doe",
                    "session_id": "sess_abcd1234",
                    "login_time": "2024-01-15T09:15:00Z",
                    "ip_address": "192.168.1.100",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                "actions": [
                    {
                        "timestamp": "2024-01-15T09:16:30Z",
                        "action": "page_view",
                        "details": {"page": "/dashboard", "duration": 45}
                    },
                    {
                        "timestamp": "2024-01-15T09:17:15Z",
                        "action": "button_click",
                        "details": {"button_id": "submit_form", "form": "user_profile"}
                    }
                ],
                "permissions": {
                    "role": "admin",
                    "access_levels": ["read", "write", "delete"],
                    "modules": {
                        "user_management": True,
                        "system_settings": True,
                        "reports": True
                    }
                }
            }
        },
        4: {
            "order_details": {
                "order_id": "ORD-12345",
                "customer": {
                    "id": "cust_789",
                    "name": "Alice Johnson",
                    "email": "alice@example.com",
                    "address": {
                        "street": "123 Main St",
                        "city": "New York",
                        "state": "NY",
                        "zip": "10001",
                        "country": "USA"
                    }
                },
                "items": [
                    {
                        "product_id": "prod_001",
                        "name": "Wireless Headphones",
                        "quantity": 2,
                        "price": 79.99,
                        "category": "Electronics",
                        "specifications": {
                            "color": "Black",
                            "wireless": True,
                            "battery_life": "20 hours"
                        }
                    },
                    {
                        "product_id": "prod_002",
                        "name": "USB Cable",
                        "quantity": 1,
                        "price": 12.99,
                        "category": "Accessories"
                    }
                ],
                "payment": {
                    "method": "credit_card",
                    "status": "completed",
                    "transaction_id": "txn_xyz789",
                    "amount": {
                        "subtotal": 172.97,
                        "tax": 13.84,
                        "shipping": 5.99,
                        "total": 192.80,
                        "currency": "USD"
                    }
                },
                "shipping": {
                    "method": "standard",
                    "estimated_delivery": "2024-01-20",
                    "tracking_number": "TRK123456789",
                    "status": "in_transit"
                }
            }
        }
    }

    # Send data to each window
    for window_num in range(1, 5):
        send_data_to_window(window_num, test_data[window_num])
        time.sleep(1)  # Wait 1 second between requests

    print("\n🔄 Sending continuous random data (Press Ctrl+C to stop)...")

    try:
        while True:
            # Pick a random window
            window_num = random.randint(1, 4)

            # Generate complex nested data based on window type
            if window_num == 1:
                data = {
                    "sensor_readings": {
                        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "location": {
                            "building": "Office Complex",
                            "floor": random.randint(1, 5),
                            "room": f"Room {random.choice(['A', 'B', 'C'])}{random.randint(1, 10)}"
                        },
                        "measurements": {
                            "temperature": {
                                "value": round(random.uniform(18.0, 28.0), 1),
                                "unit": "°C",
                                "trend": random.choice(["stable", "rising", "falling"])
                            },
                            "humidity": {
                                "value": round(random.uniform(30.0, 70.0), 1),
                                "unit": "%",
                                "status": random.choice(["normal", "high", "low"])
                            },
                            "pressure": {
                                "value": round(random.uniform(990.0, 1030.0), 2),
                                "unit": "hPa"
                            }
                        }
                    }
                }
            elif window_num == 2:
                data = {
                    "system_monitor": {
                        "server_id": f"SRV-{random.randint(100, 999)}",
                        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "performance": {
                            "cpu": {
                                "usage_percent": round(random.uniform(10.0, 90.0), 1),
                                "cores": random.choice([4, 8, 16, 32]),
                                "load_average": round(random.uniform(0.1, 4.0), 2)
                            },
                            "memory": {
                                "used_gb": round(random.uniform(2.0, 15.0), 1),
                                "total_gb": 16,
                                "available_gb": round(random.uniform(1.0, 8.0), 1)
                            },
                            "disk": {
                                "partitions": {
                                    "/": {"used": random.randint(30, 80), "total": 100, "unit": "GB"},
                                    "/var": {"used": random.randint(10, 50), "total": 50, "unit": "GB"}
                                }
                            }
                        },
                        "network": {
                            "interfaces": {
                                "eth0": {
                                    "status": "up",
                                    "ip": f"192.168.1.{random.randint(100, 200)}",
                                    "bytes_sent": random.randint(1000000, 10000000),
                                    "bytes_received": random.randint(5000000, 50000000)
                                }
                            }
                        }
                    }
                }
            elif window_num == 3:
                data = {
                    "user_analytics": {
                        "session_id": f"sess_{random.randint(100000, 999999)}",
                        "user": {
                            "id": f"user_{random.randint(1, 1000)}",
                            "role": random.choice(["admin", "user", "guest"]),
                            "location": {
                                "country": random.choice(["US", "UK", "DE", "FR"]),
                                "timezone": random.choice(["UTC", "EST", "PST", "GMT"])
                            }
                        },
                        "activity": {
                            "page_views": random.randint(1, 50),
                            "time_on_site": random.randint(60, 3600),
                            "bounce_rate": round(random.uniform(0.1, 0.8), 2),
                            "events": [
                                {
                                    "type": random.choice(["click", "scroll", "form_submit"]),
                                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                    "element": random.choice(["button", "link", "form"])
                                }
                            ]
                        },
                        "device": {
                            "type": random.choice(["desktop", "mobile", "tablet"]),
                            "browser": random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
                            "os": random.choice(["Windows", "macOS", "Linux", "iOS", "Android"])
                        }
                    }
                }
            else:  # window_num == 4
                data = {
                    "transaction": {
                        "order_id": f"ORD-{random.randint(10000, 99999)}",
                        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        "customer": {
                            "id": f"cust_{random.randint(1000, 9999)}",
                            "tier": random.choice(["bronze", "silver", "gold", "platinum"]),
                            "location": {
                                "country": random.choice(["US", "CA", "UK", "DE"]),
                                "city": random.choice(["New York", "Toronto", "London", "Berlin"])
                            }
                        },
                        "items": [
                            {
                                "product_id": f"prod_{random.randint(100, 999)}",
                                "category": random.choice(["Electronics", "Clothing", "Books", "Home"]),
                                "quantity": random.randint(1, 5),
                                "price": round(random.uniform(10.0, 500.0), 2),
                                "discount": {
                                    "applied": random.choice([True, False]),
                                    "percentage": random.randint(5, 25) if random.choice([True, False]) else 0
                                }
                            }
                        ],
                        "payment": {
                            "method": random.choice(["credit_card", "paypal", "bank_transfer"]),
                            "status": random.choice(["pending", "completed", "failed"]),
                            "currency": "USD",
                            "total": round(random.uniform(50.0, 1000.0), 2)
                        },
                        "fulfillment": {
                            "warehouse": f"WH-{random.choice(['A', 'B', 'C'])}{random.randint(1, 5)}",
                            "shipping": {
                                "method": random.choice(["standard", "express", "overnight"]),
                                "estimated_days": random.randint(1, 7)
                            }
                        }
                    }
                }

            send_data_to_window(window_num, data)
            time.sleep(random.uniform(2, 5))  # Wait 2-5 seconds between requests

    except KeyboardInterrupt:
        print("\n👋 Test client stopped")


if __name__ == "__main__":
    main()