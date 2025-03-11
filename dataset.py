# import pandas as pd
# import random

# def generate_ecommerce_dataset(num_products=100):
#     """
#     Generate a synthetic e-commerce dataset with products and their tags
    
#     Args:
#         num_products: Number of products to generate
        
#     Returns:
#         DataFrame with product information and meta tags
#     """
#     # Define possible values for different product attributes
#     categories = ["Clothing", "Footwear", "Accessories"]
    
#     clothing_types = [
#         "T-Shirt", "Hoodie", "Sweater", "Jacket", "Pants", "Jeans", "Shorts", 
#         "Dress", "Skirt", "Coat", "Shirt", "Sweatshirt", "Tank Top"
#     ]
    
#     footwear_types = [
#         "Sneakers", "Boots", "Sandals", "Loafers", "Running Shoes", "Heels", 
#         "Flats", "Slides", "Slip-ons"
#     ]
    
#     accessory_types = [
#         "Hat", "Cap", "Beanie", "Scarf", "Gloves", "Socks", "Backpack", 
#         "Bag", "Wallet", "Belt", "Sunglasses", "Jewelry", "Watch"
#     ]
    
#     materials = [
#         "Cotton", "Wool", "Polyester", "Leather", "Denim", "Silk", "Linen", 
#         "Nylon", "Canvas", "Suede", "Jersey", "Cashmere", "Fleece", "Velvet"
#     ]
    
#     colors = [
#         "Black", "White", "Red", "Blue", "Green", "Yellow", "Purple", "Pink", 
#         "Grey", "Navy", "Brown", "Orange", "Beige", "Olive", "Maroon"
#     ]
    
#     sizes = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]
    
#     styles = [
#         "Casual", "Formal", "Sporty", "Streetwear", "Vintage", "Minimalist", 
#         "Bohemian", "Preppy", "Athleisure", "Business", "Retro", "Punk", "Hipster"
#     ]
    
#     brands = [
#         "StyleCo", "UrbanThreads", "FitGear", "LuxeLiving", "OutdoorExplorers", 
#         "ModernEssentials", "ClassicWear", "TrendSetters", "ComfortFirst", 
#         "PremiumDesigns", "StreetFashion", "SportElite", "EcoChic", "FastLane", "VintageVibes"
#     ]
    
#     # Create empty list to store product data
#     products = []
    
#     # Generate unique product IDs
#     product_ids = [f"P{i:04d}" for i in range(1, num_products + 1)]
    
#     for pid in product_ids:
#         # Select category
#         category = random.choice(categories)
        
#         # Select product type based on category
#         if category == "Clothing":
#             product_type = random.choice(clothing_types)
#         elif category == "Footwear":
#             product_type = random.choice(footwear_types)
#         else:  # Accessories
#             product_type = random.choice(accessory_types)
        
#         # Select brand
#         brand = random.choice(brands)
        
#         # Set price (random between $10 and $200)
#         price = round(random.uniform(10, 200), 2)
        
#         # Select material (1-2 materials)
#         num_materials = random.randint(1, 2)
#         product_materials = random.sample(materials, num_materials)
        
#         # Select color (1-2 colors)
#         num_colors = random.randint(1, 2)
#         product_colors = random.sample(colors, num_colors)
        
#         # Select size
#         if category in ["Clothing", "Footwear"]:
#             size = random.choice(sizes)
#         else:
#             size = "OneSize"
        
#         # Select style (1-2 styles)
#         num_styles = random.randint(1, 2)
#         product_styles = random.sample(styles, num_styles)
        
#         # Create product name
#         product_name = f"{brand} {' '.join(product_materials)} {product_type} {' '.join(product_colors)}"
        
#         # Create product description
#         description = f"Premium {' & '.join(product_materials)} {product_type.lower()} from {brand}. "
#         description += f"Available in {' & '.join(product_colors).lower()}. "
#         if category in ["Clothing", "Footwear"]:
#             description += f"Size: {size}. "
#         description += f"Perfect for {' or '.join([style.lower() for style in product_styles])} style."
        
#         # Create meta tags
#         meta_tags = product_materials + [product_type] + product_colors + [size] + product_styles
        
#         # Create inventory (random between 0 and 100)
#         inventory = random.randint(0, 100)
        
#         # Add product to list
#         products.append({
#             "product_id": pid,
#             "name": product_name,
#             "category": category,
#             "product_type": product_type,
#             "brand": brand,
#             "price": price,
#             "description": description,
#             "size": size,
#             "inventory": inventory,
#             "meta": " ".join(meta_tags)
#         })
    
#     # Create DataFrame
#     df = pd.DataFrame(products)
    
#     # Ensure we have some products with common tags for better recommendation results
#     # Add some popular combinations
#     popular_combinations = [
#         {"category": "Clothing", "product_type": "Hoodie", "materials": ["Cotton"], "colors": ["Black"], "styles": ["Streetwear"]},
#         {"category": "Footwear", "product_type": "Sneakers", "materials": ["Canvas"], "colors": ["White"], "styles": ["Casual"]},
#         {"category": "Accessories", "product_type": "Backpack", "materials": ["Leather"], "colors": ["Brown"], "styles": ["Vintage"]}
#     ]
    
#     for i, combo in enumerate(popular_combinations):
#         for j in range(5):  # Add 5 products for each popular combination
#             idx = i * 5 + j
#             if idx < len(df):
#                 df.loc[idx, "category"] = combo["category"]
#                 df.loc[idx, "product_type"] = combo["product_type"]
#                 size = random.choice(sizes) if combo["category"] != "Accessories" else "OneSize"
                
#                 # Update meta tags
#                 meta_tags = combo["materials"] + [combo["product_type"]] + combo["colors"] + [size] + combo["styles"]
#                 df.loc[idx, "meta"] = " ".join(meta_tags)
                
#                 # Update name and description as well
#                 brand = df.loc[idx, "brand"]
#                 df.loc[idx, "name"] = f"{brand} {' '.join(combo['materials'])} {combo['product_type']} {' '.join(combo['colors'])}"
                
#                 description = f"Premium {' & '.join(combo['materials'])} {combo['product_type'].lower()} from {brand}. "
#                 description += f"Available in {' & '.join(combo['colors']).lower()}. "
#                 if combo["category"] in ["Clothing", "Footwear"]:
#                     description += f"Size: {size}. "
#                 description += f"Perfect for {' or '.join([style.lower() for style in combo['styles']])} style."
                
#                 df.loc[idx, "description"] = description
    
#     return df

# # Generate dataset
# products_df = generate_ecommerce_dataset(100)

# # Save to CSV file
# products_df.to_csv("ecommerce_products.csv", index=False)

# # Display the first few rows to check the data
# print(products_df.head())

import pandas as pd
import random

def generate_ecommerce_dataset(num_products=100):
    categories = ["Clothing", "Footwear", "Accessories"]
    
    clothing_types = [
        "T-Shirt", "Hoodie", "Sweater", "Jacket", "Pants", "Jeans", "Shorts", 
        "Dress", "Skirt", "Coat", "Shirt", "Sweatshirt", "Tank Top"
    ]
    
    footwear_types = [
        "Sneakers", "Boots", "Sandals", "Loafers", "Running Shoes", "Heels", 
        "Flats", "Slides", "Slip-ons"
    ]
    
    accessory_types = [
        "Hat", "Cap", "Beanie", "Scarf", "Gloves", "Socks", "Backpack", 
        "Bag", "Wallet", "Belt", "Sunglasses", "Jewelry", "Watch"
    ]
    
    materials = [
        "Cotton", "Wool", "Polyester", "Leather", "Denim", "Silk", "Linen", 
        "Nylon", "Canvas", "Suede", "Jersey", "Cashmere", "Fleece", "Velvet"
    ]
    
    colors = [
        "Black", "White", "Red", "Blue", "Green", "Yellow", "Purple", "Pink", 
        "Grey", "Navy", "Brown", "Orange", "Beige", "Olive", "Maroon"
    ]
    
    sizes = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]
    
    styles = [
        "Casual", "Formal", "Sporty", "Streetwear", "Vintage", "Minimalist", 
        "Bohemian", "Preppy", "Athleisure", "Business", "Retro", "Punk", "Hipster"
    ]
    
    brands = [
        "StyleCo", "UrbanThreads", "FitGear", "LuxeLiving", "OutdoorExplorers", 
        "ModernEssentials", "ClassicWear", "TrendSetters", "ComfortFirst", 
        "PremiumDesigns", "StreetFashion", "SportElite", "EcoChic", "FastLane", "VintageVibes"
    ]
    
    products = []
    product_ids = [f"P{i:04d}" for i in range(1, num_products + 1)]
    
    for pid in product_ids:
        category = random.choice(categories)
        
        if category == "Clothing":
            product_type = random.choice(clothing_types)
        elif category == "Footwear":
            product_type = random.choice(footwear_types)
        else:
            product_type = random.choice(accessory_types)
        
        brand = random.choice(brands)
        price = round(random.uniform(10, 200), 2)
        num_materials = random.randint(1, 2)
        product_materials = random.sample(materials, num_materials)
        num_colors = random.randint(1, 2)
        product_colors = random.sample(colors, num_colors)
        size = random.choice(sizes) if category in ["Clothing", "Footwear"] else "OneSize"
        num_styles = random.randint(1, 2)
        product_styles = random.sample(styles, num_styles)
        
        product_name = f"{brand} {' '.join(product_materials)} {product_type} {' '.join(product_colors)}"
        description = f"Premium {' & '.join(product_materials)} {product_type.lower()} from {brand}. "
        description += f"Available in {' & '.join(product_colors).lower()}. "
        if category in ["Clothing", "Footwear"]:
            description += f"Size: {size}. "
        description += f"Perfect for {' or '.join([style.lower() for style in product_styles])} style."
        
        meta_tags = product_materials + [product_type] + product_colors + [size] + product_styles
        inventory = random.randint(0, 100)
        
        products.append({
            "product_id": pid,
            "name": product_name,
            "category": category,
            "product_type": product_type,
            "brand": brand,
            "price": price,
            "description": description,
            "size": size,
            "inventory": inventory,
            "meta": " ".join(meta_tags)
        })
    
    df = pd.DataFrame(products)
    
    popular_combinations = [
        {"category": "Clothing", "product_type": "Hoodie", "materials": ["Cotton"], "colors": ["Black"], "styles": ["Streetwear"]},
        {"category": "Footwear", "product_type": "Sneakers", "materials": ["Canvas"], "colors": ["White"], "styles": ["Casual"]},
        {"category": "Accessories", "product_type": "Backpack", "materials": ["Leather"], "colors": ["Brown"], "styles": ["Vintage"]}
    ]
    
    for i, combo in enumerate(popular_combinations):
        for j in range(5):
            idx = i * 5 + j
            if idx < len(df):
                df.loc[idx, "category"] = combo["category"]
                df.loc[idx, "product_type"] = combo["product_type"]
                size = random.choice(sizes) if combo["category"] != "Accessories" else "OneSize"
                meta_tags = combo["materials"] + [combo["product_type"]] + combo["colors"] + [size] + combo["styles"]
                df.loc[idx, "meta"] = " ".join(meta_tags)
                brand = df.loc[idx, "brand"]
                df.loc[idx, "name"] = f"{brand} {' '.join(combo['materials'])} {combo['product_type']} {' '.join(combo['colors'])}"
                description = f"Premium {' & '.join(combo['materials'])} {combo['product_type'].lower()} from {brand}. "
                description += f"Available in {' & '.join(combo['colors']).lower()}. "
                if combo["category"] in ["Clothing", "Footwear"]:
                    description += f"Size: {size}. "
                description += f"Perfect for {' or '.join([style.lower() for style in combo['styles']])} style."
                df.loc[idx, "description"] = description
    
    return df

products_df = generate_ecommerce_dataset(100)
products_df.to_csv("ecommerce_products.csv", index=False)
print(products_df.head())
